// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

namespace onnxruntime {
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
  explicit LayeringRuleMatcher(const LayeringRules& rules) {
    for (size_t i = 0; i < rules.rules.size(); ++i) {
      const auto& rule = rules.rules[i];
      if (rule.prefix_match) {
        AddPrefixRule(rule.annotation, i);
      } else {
        AddExactRule(rule.annotation, i);
      }
    }
  }

  /// <summary>
  /// The method returns the index of the best matching rule for the given annotation
  /// if it exists
  /// </summary>
  /// <param name="node_annotation">annotation retrieved from protobuf node metadata</param>
  /// <returns></returns>
  std::optional<size_t> Match(const std::string& node_annotation) const {
    std::optional<size_t> best_match = std::nullopt;

    // 1. Check Exact Matches
    auto it = exact_match_rules_.find(node_annotation);
    if (it != exact_match_rules_.end()) {
      best_match = it->second;
    }

    // 2. Check Prefix Matches via Trie
    const TrieNode* current = &root_;

    // Check for empty prefix rule (matches everything)
    if (current->rule_index.has_value()) {
      UpdateBestMatch(best_match, *current->rule_index);
    }

    for (char c : node_annotation) {
      if (best_match.has_value() && *best_match == 0) {
        // Optimization: If we already found index 0, we can't do better.
        return 0;
      }

      auto child_it = current->children.find(c);
      if (child_it == current->children.end()) {
        break;
      }
      current = child_it->second.get();
      if (current->rule_index.has_value()) {
        UpdateBestMatch(best_match, *current->rule_index);
      }
    }

    return best_match;
  }

 private:
  struct TrieNode {
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    std::optional<size_t> rule_index;
  };

  TrieNode root_;
  std::unordered_map<std::string, size_t> exact_match_rules_;

  void AddExactRule(const std::string& annotation, size_t index) {
    // Only store the first occurrence (lowest index)
    if (exact_match_rules_.find(annotation) == exact_match_rules_.end()) {
      exact_match_rules_[annotation] = index;
    }
  }

  void AddPrefixRule(const std::string& annotation, size_t index) {
    TrieNode* current = &root_;
    for (char c : annotation) {
      if (current->children.find(c) == current->children.end()) {
        current->children[c] = std::make_unique<TrieNode>();
      }
      current = current->children[c].get();
    }

    // Only store if strictly better (lower index) or not set
    if (!current->rule_index.has_value() || index < *current->rule_index) {
      current->rule_index = index;
    }
  }

  void UpdateBestMatch(std::optional<size_t>& current_best, size_t candidate) const {
    if (!current_best.has_value() || candidate < *current_best) {
      current_best = candidate;
    }
  }
};

}
