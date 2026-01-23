// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/common/narrow.h"
#include "core/common/string_utils.h"
#include "core/framework/layering_annotations.h"
#include "core/framework/ortmemoryinfo.h"
#include "core/session/abi_devices.h"

#include <limits>

namespace onnxruntime {

LayeringRules LayeringRules::FromConfigString(const std::string& config_value) {
  LayeringRules rules;
  if (config_value.empty()) {
    return rules;
  }

  auto entries = utils::SplitString(config_value, ";");
  for (const auto& e : entries) {
    auto entry = utils::TrimString(e);
    if (entry.empty()) {
      continue;
    }

    const size_t open_paren = entry.find('(');
    const size_t close_paren = entry.find(')');

    if (open_paren == std::string::npos || close_paren == std::string::npos || close_paren < open_paren) {
      continue;
    }

    std::string device = entry.substr(0, open_paren);
    device = utils::TrimString(device);

    if (device.empty()) {
      continue;
    }

    std::string annotations_list = entry.substr(open_paren + 1, close_paren - open_paren - 1);
    auto annotations = utils::SplitString(annotations_list, ",");
    for (auto& a : annotations) {
      auto ann = utils::TrimString(a);
      if (ann.empty()) {
        continue;
      }

      bool prefix_match = false;
      if (ann[0] == '=') {
        prefix_match = true;
        ann = ann.substr(1);
        ann = utils::TrimString(ann);
      }

      if (ann.empty()) {
        continue;
      }

      rules.rules.push_back({device, std::move(ann), prefix_match});
    }
  }

  return rules;
}

LayeringRuleMatcher::LayeringRuleMatcher(const LayeringRules& rules) {
  for (size_t i = 0; i < rules.rules.size(); ++i) {
    const auto& rule = rules.rules[i];
    ORT_ENFORCE(!rule.annotation.empty(), "Layering rule annotation cannot be empty");
    if (rule.prefix_match) {
      AddPrefixRule(rule.annotation, i);
    } else {
      AddExactRule(rule.annotation, i);
    }
  }
}

std::optional<size_t> LayeringRuleMatcher::Match(const std::string& node_annotation) const {
  std::optional<size_t> best_match = std::nullopt;

  // 1. Check Prefix Matches via Trie. Prefix have priority over exact matches.
  const TrieNode* current = &root_;

  // No empty annotations
  // so we omit checking the root.

  for (char c : node_annotation) {
    if (best_match && *best_match == 0) {
      // Optimization: If we already found index 0, we can't do better.
      return best_match;
    }

    auto child_it = current->children.find(c);
    if (child_it == current->children.end()) {
      break;
    }
    current = child_it->second.get();
    if (current->rule_index) {
      UpdateBestMatch(best_match, *current->rule_index);
    }
  }

  if (best_match) {
    return best_match;
  }

  // 2. Check Exact Matches (fallback)
  auto it = exact_match_rules_.find(node_annotation);
  if (it != exact_match_rules_.end()) {
    best_match = it->second;
  }

  return best_match;
}

namespace {
bool CaseInsensitiveCompare(std::string_view a, std::string_view b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                    [](char c1, char c2) { return std::tolower(c1) == std::tolower(c2); });
}

bool TryParseIndex(const std::string& str, uint32_t& index) {
  char* end = nullptr;
  const char* ptr = str.c_str();
  errno = 0;
  unsigned long val = std::strtoul(ptr, &end, 10);
  if (errno != 0 || end != ptr + str.size()) {
    return false;
  }
  index = narrow<uint32_t>(val);
  return true;
}
}  // namespace

std::optional<EpMatchResult> EpLayeringMatcher::Match(const OrtEpDevice& ep_device, const LayeringRules& rules,
                                                      size_t rule_index) {
  if (rule_index >= rules.rules.size()) {
    return std::nullopt;
  }

  const std::string& target_full = rules.rules[rule_index].device;
  const auto colon_pos = target_full.find(':');
  const std::string target_type_str = (colon_pos == std::string::npos) ? target_full : target_full.substr(0, colon_pos);
  // vendor or index or uuid, if present
  std::string target_specifier;
  if (colon_pos != std::string::npos) {
    target_specifier = target_full.substr(colon_pos + 1);
  }

  bool matched = false;

  // Helper to check device type from MemInfo if Hardware device logic fails/is absent
  auto check_mem_device_type = [&](OrtDevice::DeviceType type) -> bool {
    if (ep_device.device_memory_info) {
      return ep_device.device_memory_info->device.Type() == type;
    }
    return false;
  };

  // 1. Exact Name / Alias match
  // "cpu"
  if (CaseInsensitiveCompare(target_type_str, "cpu")) {
    if (ep_device.ep_name == kCpuExecutionProvider) {
      matched = true;
    } else if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_CPU) {
      matched = true;
    } else if (check_mem_device_type(OrtDevice::CPU)) {
      matched = true;
    }
  }  // "gpu"
  else if (CaseInsensitiveCompare(target_type_str, "gpu")) {
    // If simple "gpu"
    if (target_specifier.empty()) {
      if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_GPU) {
        matched = true;
      } else if (check_mem_device_type(OrtDevice::GPU)) {
        matched = true;
      }  // Heuristic fallback for common GPU EPs if hardware info is missing
      else if (ep_device.ep_name == kCudaExecutionProvider || ep_device.ep_name == kDmlExecutionProvider) {
        matched = true;
      }
    } else {
      // "gpu:<vendor>" or "gpu:<index>"
      if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_GPU) {
        uint32_t index = std::numeric_limits<uint32_t>::max();
        if (TryParseIndex(target_specifier, index)) {
          // gpu:<index>
          if (ep_device.device->device_id == index) {
            matched = true;
          }
        } else {
          // gpu:<vendor>
          if (CaseInsensitiveCompare(ep_device.device->vendor, target_specifier)) {
            matched = true;
          }
          // Check against vendor ID
          else if (CaseInsensitiveCompare(target_specifier, "nvidia") && ep_device.device->vendor_id == OrtDevice::VendorIds::NVIDIA) {
            matched = true;
          } else if (CaseInsensitiveCompare(target_specifier, "amd") && ep_device.device->vendor_id == OrtDevice::VendorIds::AMD) {
            matched = true;
          } else if (CaseInsensitiveCompare(target_specifier, "intel") && ep_device.device->vendor_id == OrtDevice::VendorIds::INTEL) {
            matched = true;
          }
          // Special shortcuts heuristics: gpu:nvidia -> CUDA
          else if (CaseInsensitiveCompare(target_specifier, "nvidia") && ep_device.ep_name == kCudaExecutionProvider) {
            matched = true;
          }
        }
      }
    }
  }
  // "accelerator" (not cpu)
  else if (CaseInsensitiveCompare(target_type_str, "accelerator")) {
    if (ep_device.ep_name != kCpuExecutionProvider) {
      // If we don't have HW info, assuming non-CPU EP is an accelerator.
      // If we do have HW info, check it's not CPU.
      const bool is_cpu_hw = (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_CPU);
      const bool is_cpu_mem = check_mem_device_type(OrtDevice::CPU);

      if (!is_cpu_hw && !is_cpu_mem) {
        matched = true;
      }
    }
  }  // "npu"
  else if (CaseInsensitiveCompare(target_type_str, "npu")) {
    if (ep_device.device && ep_device.device->type == OrtHardwareDeviceType_NPU) {
      matched = true;
    } else if (ep_device.ep_name == kQnnExecutionProvider || ep_device.ep_name == kVitisAIExecutionProvider) {
      // Heuristic for known NPU providers if HW device info is missing
      // XXX: These can run on CPU as well, need to see if there any check that is missing.
      matched = true;
    }
  }
  // "fpga"
  else if (CaseInsensitiveCompare(target_type_str, "fpga")) {
    // No OrtHardwareDeviceType_FPGA currently, rely on OrtDevice::FPGA from MemInfo
    if (check_mem_device_type(OrtDevice::FPGA)) {
      matched = true;
    }
  }
  // "cuda"
  else if (CaseInsensitiveCompare(target_type_str, "cuda")) {
    if (ep_device.ep_name == kCudaExecutionProvider) {
      matched = true;
    }
  }
  // "dml"
  else if (CaseInsensitiveCompare(target_type_str, "dml")) {
    if (ep_device.ep_name == kDmlExecutionProvider) {
      matched = true;
    }
  }
  // Fallback: Exact EP name string match (e.g. "MyCustomEP")
  else if (ep_device.ep_name == target_full) {
    matched = true;
  }

  if (matched) {
    return EpMatchResult{rule_index, ep_device.ep_name};
  }

  return std::nullopt;
}

}  // namespace onnxruntime