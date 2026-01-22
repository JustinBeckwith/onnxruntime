// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/layering_annotations.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(LayeringRuleMatcherTest, ExactMatches) {
  LayeringRules rules;
  rules.rules.push_back({"Device1", "Annotation1", false});  // Index 0
  rules.rules.push_back({"Device2", "Annotation2", false});  // Index 1

  LayeringRuleMatcher matcher(rules);

  {
    auto result = matcher.Match("Annotation1");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }
  {
    auto result = matcher.Match("Annotation2");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1u);
  }
  {
    auto result = matcher.Match("Annotation3");
    EXPECT_FALSE(result.has_value());
  }
}

TEST(LayeringRuleMatcherTest, PrefixMatches) {
  LayeringRules rules;
  rules.rules.push_back({"Device1", "Prefix1", true});  // Index 0: =Prefix1
  rules.rules.push_back({"Device2", "Pre", true});      // Index 1: =Pre

  LayeringRuleMatcher matcher(rules);

  // "Prefix1Suffix" matches "Prefix1" (idx 0) and "Pre" (idx 1). 0 < 1, so 0.
  {
    auto result = matcher.Match("Prefix1Suffix");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }

  // "PreSuffix" matches "Pre" (idx 1). "Prefix1" does not match.
  {
    auto result = matcher.Match("PreSuffix");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1u);
  }

  // "Other" matches nothing
  {
    auto result = matcher.Match("Other");
    EXPECT_FALSE(result.has_value());
  }
}

TEST(LayeringRuleMatcherTest, PriorityPrefixOverExact) {
  // Prefix matches should take precedence over exact matches regardless of order.

  // Case 1: Prefix rule comes before Exact rule
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "A", true});    // Index 0: =A (Prefix)
    rules.rules.push_back({"Device2", "AB", false});  // Index 1: AB (Exact)

    LayeringRuleMatcher matcher(rules);
    // "AB" matches prefix "A" (idx 0) and exact "AB" (idx 1).
    // Since prefix matches are checked first and returned if found, we expect 0.
    auto result = matcher.Match("AB");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }

  // Case 2: Exact rule comes before Prefix rule
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "AB", false});  // Index 0: AB (Exact)
    rules.rules.push_back({"Device2", "A", true});    // Index 1: =A (Prefix)

    LayeringRuleMatcher matcher(rules);
    // "AB" matches exact "AB" (idx 0) and prefix "A" (idx 1).
    // Priority says Prefix matches are returned first.
    auto result = matcher.Match("AB");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1u);
  }
}

TEST(LayeringRuleMatcherTest, LongestOrShortestPrefixPriority) {
  // If multiple prefix rules match, the one with the lowest index (earliest in config) wins.

  // Case 1: Shorter prefix first
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "A", true});   // Index 0
    rules.rules.push_back({"Device2", "AB", true});  // Index 1

    LayeringRuleMatcher matcher(rules);
    // "ABC" matches "A" (0) and "AB" (1). Since 0 < 1, best match is 0.
    auto result = matcher.Match("ABC");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }

  // Case 2: Longer prefix first
  {
    LayeringRules rules;
    rules.rules.push_back({"Device1", "AB", true});  // Index 0
    rules.rules.push_back({"Device2", "A", true});   // Index 1

    LayeringRuleMatcher matcher(rules);
    // "ABC" matches "AB" (0) and "A" (1). Since 0 < 1, best match is 0.
    auto result = matcher.Match("ABC");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0u);
  }
}

TEST(LayeringRuleMatcherTest, OverlappingExactMatchPriority) {
  // If duplicates exist, first one wins.
  LayeringRules rules;
  rules.rules.push_back({"Device1", "A", false});  // Index 0
  rules.rules.push_back({"Device2", "A", false});  // Index 1

  LayeringRuleMatcher matcher(rules);
  auto result = matcher.Match("A");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 0u);
}

TEST(LayeringRuleMatcherTest, OverlappingPrefixMatchPriority) {
  // If duplicates exist, first one wins.
  LayeringRules rules;
  rules.rules.push_back({"Device1", "A", true});  // Index 0
  rules.rules.push_back({"Device2", "A", true});  // Index 1

  LayeringRuleMatcher matcher(rules);
  auto result = matcher.Match("AB");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 0u);
}

}  // namespace test
}  // namespace onnxruntime
