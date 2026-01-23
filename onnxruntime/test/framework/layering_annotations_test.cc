// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ortmemoryinfo.h"
#include "core/framework/layering_annotations.h"
#include "core/session/abi_devices.h"
#include "core/graph/constants.h"
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

namespace {

// Helper to construct OrtEpDevice wrappers for testing
struct TestEpDevice {
  std::string ep_name;
  OrtHardwareDevice hw_device;
  bool has_hw_device = false;
  OrtMemoryInfo mem_info;
  bool has_mem_info = false;

  // We need to keep the structures alive while OrtEpDevice points to them
  OrtEpDevice Get() const {
    OrtEpDevice ep;
    ep.ep_name = ep_name;
    ep.device = has_hw_device ? &hw_device : nullptr;
    ep.device_memory_info = has_mem_info ? &mem_info : nullptr;
    return ep;
  }
};

TestEpDevice CreateEp(const std::string& name) {
  TestEpDevice ep;
  ep.ep_name = name;
  return ep;
}

TestEpDevice CreateHwEp(const std::string& name, OrtHardwareDeviceType type, uint32_t vendor_id = 0,
                        uint32_t device_id = 0, const std::string& vendor_str = std::string()) {
  TestEpDevice ep;
  ep.ep_name = name;
  ep.hw_device = {type, vendor_id, device_id, vendor_str, {}};
  ep.has_hw_device = true;
  return ep;
}

TestEpDevice CreateMemEp(const std::string& name, OrtDevice::DeviceType type, int device_id = 0) {
  TestEpDevice ep;
  ep.ep_name = name;
  // Note: OrtMemoryInfo name doesn't matter for logic now, but required for ctor
  ep.mem_info = OrtMemoryInfo("TestMem", OrtAllocatorType::OrtDeviceAllocator,
                              OrtDevice(type, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE,
                                        static_cast<OrtDevice::DeviceId>(device_id)),
                              OrtMemType::OrtMemTypeDefault);
  ep.has_mem_info = true;
  return ep;
}

}  // namespace

TEST(EpLayeringMatcherTest, MatchCPU) {
  LayeringRules rules;
  rules.rules.push_back({"CPU", "Anno1", false});  // Index 0

  // Case 1: EP Name kCpuExecutionProvider
  {
    auto test_ep = CreateEp(kCpuExecutionProvider);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->ep_type, kCpuExecutionProvider);
  }

  // Case 2: Hardware Device CPU
  {
    auto test_ep = CreateHwEp("SomeCPU_EP", OrtHardwareDeviceType_CPU);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 3: Memory Info CPU
  {
    auto test_ep = CreateMemEp("MemCPU_EP", OrtDevice::CPU);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchGPU) {
  LayeringRules rules;
  rules.rules.push_back({"GPU", "Anno1", false});  // Index 0

  // Case 1: Hardware Device GPU
  {
    auto test_ep = CreateHwEp("MyGPU_EP", OrtHardwareDeviceType_GPU);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 2: Memory Info GPU
  {
    auto test_ep = CreateMemEp("MemGPU_EP", OrtDevice::GPU);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 3: Heuristic kCudaExecutionProvider
  {
    auto test_ep = CreateEp(kCudaExecutionProvider);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 4: Heuristic kDmlExecutionProvider
  {
    auto test_ep = CreateEp(kDmlExecutionProvider);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_VendorString) {
  LayeringRules rules;
  rules.rules.push_back({"gpu:nvidia", "Anno1", false});  // Index 0

  // Case 1: Vendor String Match
  {
    auto test_ep = CreateHwEp("MyNvidia_EP", OrtHardwareDeviceType_GPU, 0, 0, "NVIDIA");
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 2: Vendor String Mismatch
  {
    auto test_ep = CreateHwEp("MyAMD_EP", OrtHardwareDeviceType_GPU, 0, 0, "AMD");
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    EXPECT_FALSE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_VendorId) {
  LayeringRules rules;
  rules.rules.push_back({"gpu:intel", "Anno1", false});   // Index 0
  rules.rules.push_back({"gpu:nvidia", "Anno2", false});  // Index 1
  rules.rules.push_back({"gpu:amd", "Anno3", false});     // Index 2

  // Case 1: Vendor ID Match Intel
  {
    auto test_ep = CreateHwEp("Intel_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::INTEL);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rule_index, 0u);
  }

  // Case 2: Vendor ID Match Nvidia
  {
    auto test_ep = CreateHwEp("Nvidia_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::NVIDIA);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 1);
    ASSERT_TRUE(result.has_value());
  }

  // Case 3: Vendor ID Match AMD
  {
    auto test_ep = CreateHwEp("AMD_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::AMD);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 2);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_Heuristic) {
  LayeringRules rules;
  rules.rules.push_back({"gpu:nvidia", "Anno1", false});  // Index 0

  // Case 1: kCudaExecutionProvider -> nvidia
  {
    auto test_ep = CreateEp(kCudaExecutionProvider);
    // Needs HW Type GPU for this specific heuristic path in code?
    // Looking at code: "else if (target_specifier == "nvidia" && ep_name == kCuda)" lies inside "if (device && type == GPU)".
    // So heuristic applies ONLY if we have HW info saying it is a GPU.

    // Let's create an EP that claims to be a GPU HW but has generic/empty vendor
    auto test_ep_hw = CreateHwEp(kCudaExecutionProvider, OrtHardwareDeviceType_GPU);

    auto result = EpLayeringMatcher::Match(test_ep_hw.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_Index) {
  LayeringRules rules;
  rules.rules.push_back({"gpu:1", "Anno1", false});  // Index 0

  // Case 1: ID Match
  {
    auto test_ep = CreateHwEp("GPU1", OrtHardwareDeviceType_GPU, 0, 1);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 2: ID Mismatch
  {
    auto test_ep = CreateHwEp("GPU0", OrtHardwareDeviceType_GPU, 0, 0);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    EXPECT_FALSE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchAccelerator) {
  LayeringRules rules;
  rules.rules.push_back({"accelerator", "Anno1", false});  // Index 0

  // Case 1: CPU EP should NOT match
  {
    auto test_ep = CreateEp(kCpuExecutionProvider);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    EXPECT_FALSE(result.has_value());
  }

  // Case 2: Custom EP, No HW/Mem info, considered accelerator
  {
    auto test_ep = CreateEp("MyCustomAccel");
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 3: GPU HW is an accelerator
  {
    auto test_ep = CreateHwEp("MyGPU", OrtHardwareDeviceType_GPU);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchNPU) {
  LayeringRules rules;
  rules.rules.push_back({"npu", "Anno1", false});  // Index 0

  // Case 1: Hardware NPU
  {
    auto test_ep = CreateHwEp("MyNPU", OrtHardwareDeviceType_NPU);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }

  // Case 2: QNN Heuristic
  {
    auto test_ep = CreateEp(kQnnExecutionProvider);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchFPGA) {
  LayeringRules rules;
  rules.rules.push_back({"fpga", "Anno1", false});  // Index 0

  // Case 1: MemInfo says FPGA
  {
    auto test_ep = CreateMemEp("MyFPGA", OrtDevice::FPGA);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchDirectDesignators) {
  LayeringRules rules;
  rules.rules.push_back({"cuda", "A", false});  // 0
  rules.rules.push_back({"dml", "B", false});   // 1

  {
    auto test_ep = CreateEp(kCudaExecutionProvider);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
  {
    auto test_ep = CreateEp(kDmlExecutionProvider);
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 1);
    ASSERT_TRUE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchExactEPName) {
  LayeringRules rules;
  rules.rules.push_back({"MyCustomEP", "Anno1", false});

  {
    auto test_ep = CreateEp("MyCustomEP");
    auto result = EpLayeringMatcher::Match(test_ep.Get(), rules, 0);
    ASSERT_TRUE(result.has_value());
  }
}

}  // namespace test
}  // namespace onnxruntime
