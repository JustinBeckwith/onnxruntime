// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "gsl/gsl"
#include <string>
#include <vector>
#include <optional>

struct OrtEpDevice;

namespace onnxruntime {
class ExecutionProviders;

/// <summary>
/// Annotation extracted from kOrtSessionOptionsLayerAssignmentSettings session configuration option.
/// </summary>
struct LayerAnnotation {
  std::string device;
  std::string annotation;
  bool prefix_match;
};

/// <summary>
/// This struct is a container for layering rules extracted from the kOrtSessionOptionsLayerAssignmentSettings
/// session configuration option.
/// </summary>
struct LayeringRules {
  std::vector<LayerAnnotation> rules;
  /// <summary>
  /// Parses the layering rules from the given configuration string.
  /// </summary>
  /// <param name="config_value"></param>
  /// <returns></returns>
  static LayeringRules FromConfigString(const std::string& config_value);
};

/// <summary>
/// This class matches node annotations against layering rules.
/// </summary>
class LayeringRuleMatcher {
 public:
  explicit LayeringRuleMatcher(const LayeringRules& rules);

  /// <summary>
  /// The method returns the index of the best matching rule for the given annotation
  /// if it exists
  /// </summary>
  /// <param name="node_annotation">annotation retrieved from protobuf node metadata</param>
  /// <returns>index of the matching LayeringRule if it exists</returns>
  std::optional<size_t> Match(const std::string& node_annotation) const;

 private:
  struct TrieNode {
    InlinedHashMap<char, std::unique_ptr<TrieNode>> children;
    std::optional<size_t> rule_index;
  };

  TrieNode root_;
  InlinedHashMap<std::string, size_t> exact_match_rules_;

  void AddExactRule(const std::string& annotation, size_t index) {
    // Only store the first occurrence (lowest index)
    exact_match_rules_.insert({annotation, index});
  }

  void AddPrefixRule(const std::string& annotation, size_t index) {
    TrieNode* current = &root_;
    for (char c : annotation) {
      auto p = current->children.insert({c, nullptr});
      if (p.second) {
        p.first->second = std::make_unique<TrieNode>();
      }
      current = p.first->second.get();
    }

    // Only store if strictly better (lower index) or not set
    // Since we iterate rules 0..N, if a rule index is already set for this node,
    // it corresponds to a higher priority rule, so we skip overwriting it.
    if (!current->rule_index) {
      current->rule_index = index;
    }
  }

  void UpdateBestMatch(std::optional<size_t>& current_best, size_t candidate) const {
    if (!current_best || candidate < *current_best) {
      current_best = candidate;
    }
  }
};

class EpLayeringMatcher {
 public:
  /// <summary>
  /// Matches a list of available OrtEpDevices against the device string specified in the LayerAnnotation.
  /// Returns the EP Type string of the first device that matches the rule.
  /// </summary>
  /// <param name="ep_devices">The list of available EP devices.</param>
  /// <param name="rule">The rule containing the device designator.</param>
  /// <returns>Optional containing the matched EP type, nullopt otherwise.</returns>
  static std::optional<std::string> Match(gsl::span<const OrtEpDevice* const> ep_devices,
                                          const LayerAnnotation& rule);

  /// <summary>
  /// Matches a collection of ExecutionProviders against the device string specified in the LayerAnnotation.
  /// Returns the EP Type string of the first provider that matches the rule.
  /// </summary>
  /// <param name="providers">The collection of available Execution Providers.</param>
  /// <param name="rule">The rule containing the device designator.</param>
  /// <returns>Optional containing the matched EP type, nullopt otherwise.</returns>
  static std::optional<std::string> Match(const ExecutionProviders& providers, const LayerAnnotation& rule);
};

}  // namespace onnxruntime
