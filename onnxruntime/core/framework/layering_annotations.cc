// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layering_annotations.h"
#include "core/common/string_utils.h"

namespace onnxruntime {
LayeringRules LayeringRules::FromConfigString(const std::string& config_value) {
  LayeringRules rules;
  if (config_value.empty()) {
    return rules;
  }

  auto trim = [](std::string& s) {
    if (s.empty()) return;
    s.erase(0, s.find_first_not_of(" \t\n\r"));
    s.erase(s.find_last_not_of(" \t\n\r") + 1);
  };

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

}  // namespace onnxruntime