// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ortmemoryinfo.h"
#include "core/framework/layering_annotations.h"
#include "core/session/abi_devices.h"
#include "core/framework/execution_provider.h"  // For kCpuExecutionProvider, kCudaExecutionProvider, etc.
#include "core/framework/ortdevice.h"
#include "core/graph/constants.h"
#include "gtest/gtest.h"
#include "core/framework/execution_providers.h"  // For ExecutionProviders

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
  LayerAnnotation rule = {"CPU", "Anno1", false};

  // Case 1: EP Name kCpuExecutionProvider
  {
    auto test_ep = CreateEp(kCpuExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCpuExecutionProvider);
  }

  // Case 2: Hardware Device CPU
  {
    auto test_ep = CreateHwEp("SomeCPU_EP", OrtHardwareDeviceType_CPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "SomeCPU_EP");
  }

  // Case 3: Memory Info CPU
  {
    auto test_ep = CreateMemEp("MemCPU_EP", OrtDevice::CPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MemCPU_EP");
  }
}

TEST(EpLayeringMatcherTest, MatchGPU) {
  LayerAnnotation rule = {"GPU", "Anno1", false};

  // Case 1: Hardware Device GPU
  {
    auto test_ep = CreateHwEp("MyGPU_EP", OrtHardwareDeviceType_GPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyGPU_EP");
  }

  // Case 2: Memory Info GPU
  {
    auto test_ep = CreateMemEp("MemGPU_EP", OrtDevice::GPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MemGPU_EP");
  }

  // Case 3: Heuristic kCudaExecutionProvider
  {
    auto test_ep = CreateEp(kCudaExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCudaExecutionProvider);
  }

  // Case 4: Heuristic kDmlExecutionProvider
  {
    auto test_ep = CreateEp(kDmlExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kDmlExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_VendorString) {
  LayerAnnotation rule = {"gpu:nvidia", "Anno1", false};

  // Case 1: Vendor String Match
  {
    auto test_ep = CreateHwEp("MyNvidia_EP", OrtHardwareDeviceType_GPU, 0, 0, "NVIDIA");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyNvidia_EP");
  }

  // Case 2: Vendor String Mismatch
  {
    auto test_ep = CreateHwEp("MyAMD_EP", OrtHardwareDeviceType_GPU, 0, 0, "AMD");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    EXPECT_FALSE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_VendorId) {
  LayerAnnotation rule_intel = {"gpu:intel", "Anno1", false};
  LayerAnnotation rule_nvidia = {"gpu:nvidia", "Anno2", false};
  LayerAnnotation rule_amd = {"gpu:amd", "Anno3", false};

  // Case 1: Vendor ID Match Intel
  {
    auto test_ep = CreateHwEp("Intel_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::INTEL);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_intel);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Intel_EP");
  }

  // Case 2: Vendor ID Match Nvidia
  {
    auto test_ep = CreateHwEp("Nvidia_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::NVIDIA);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_nvidia);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Nvidia_EP");
  }

  // Case 3: Vendor ID Match AMD
  {
    auto test_ep = CreateHwEp("AMD_EP", OrtHardwareDeviceType_GPU, OrtDevice::VendorIds::AMD);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_amd);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "AMD_EP");
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_Heuristic) {
  LayerAnnotation rule = {"gpu:nvidia", "Anno1", false};

  // Case 1: kCudaExecutionProvider -> nvidia
  {
    // Need an EP with GPU HW type but generic vendor info to trigger the heuristic
    auto test_ep_hw = CreateHwEp(kCudaExecutionProvider, OrtHardwareDeviceType_GPU);
    OrtEpDevice ep_device = test_ep_hw.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};

    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCudaExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchSpecificGPU_Index) {
  LayerAnnotation rule = {"gpu:1", "Anno1", false};

  // Case 1: ID Match
  {
    auto test_ep = CreateHwEp("GPU1", OrtHardwareDeviceType_GPU, 0, 1);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};

    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "GPU1");
  }

  // Case 2: ID Mismatch
  {
    auto test_ep = CreateHwEp("GPU0", OrtHardwareDeviceType_GPU, 0, 0);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    EXPECT_FALSE(result.has_value());
  }
}

TEST(EpLayeringMatcherTest, MatchAccelerator) {
  LayerAnnotation rule = {"accelerator", "Anno1", false};

  // Case 1: CPU EP should NOT match
  {
    auto test_ep = CreateEp(kCpuExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    EXPECT_FALSE(result.has_value());
  }

  // Case 2: Custom EP, No HW/Mem info, considered accelerator
  {
    auto test_ep = CreateEp("MyCustomAccel");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyCustomAccel");
  }

  // Case 3: GPU HW is an accelerator
  {
    auto test_ep = CreateHwEp("MyGPU", OrtHardwareDeviceType_GPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyGPU");
  }
}

TEST(EpLayeringMatcherTest, MatchNPU) {
  LayerAnnotation rule = {"npu", "Anno1", false};

  // Case 1: Hardware NPU
  {
    auto test_ep = CreateHwEp("MyNPU", OrtHardwareDeviceType_NPU);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyNPU");
  }

  // Case 2: QNN Heuristic
  {
    auto test_ep = CreateEp(kQnnExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kQnnExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchFPGA) {
  LayerAnnotation rule = {"fpga", "Anno1", false};

  // Case 1: MemInfo says FPGA
  {
    auto test_ep = CreateMemEp("MyFPGA", OrtDevice::FPGA);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyFPGA");
  }
}

TEST(EpLayeringMatcherTest, MatchDirectDesignators) {
  LayerAnnotation rule_cuda = {"cuda", "A", false};
  LayerAnnotation rule_dml = {"dml", "B", false};

  {
    auto test_ep = CreateEp(kCudaExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_cuda);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kCudaExecutionProvider);
  }
  {
    auto test_ep = CreateEp(kDmlExecutionProvider);
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule_dml);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, kDmlExecutionProvider);
  }
}

TEST(EpLayeringMatcherTest, MatchExactEPName) {
  LayerAnnotation rule = {"MyCustomEP", "Anno1", false};

  {
    auto test_ep = CreateEp("MyCustomEP");
    OrtEpDevice ep_device = test_ep.Get();
    std::vector<const OrtEpDevice*> devices = {&ep_device};
    auto result = EpLayeringMatcher::Match(devices, rule);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "MyCustomEP");
  }
}

namespace {

// Minimal concrete implementation of IExecutionProvider for testing
class MockExecutionProvider : public IExecutionProvider {
 public:
  MockExecutionProvider(const std::string& type, OrtDevice device)
      : IExecutionProvider(type, device) {}

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override { return nullptr; }
};

}  // namespace

TEST(EpLayeringMatcherTest, MatchExecutionProviders_CPU) {
  LayerAnnotation rule = {"CPU", "Anno1", false};
  ExecutionProviders providers;

  // Add CPU provider
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add a GPU provider (should be skipped for CPU rule)
  auto gpu_ep = std::make_shared<MockExecutionProvider>(kCudaExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCudaExecutionProvider, gpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, kCpuExecutionProvider);
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_GPU) {
  LayerAnnotation rule = {"GPU", "Anno1", false};
  ExecutionProviders providers;

  // Add CPU provider (should be skipped)
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add CUDA provider (GPU)
  auto gpu_ep = std::make_shared<MockExecutionProvider>(kCudaExecutionProvider, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCudaExecutionProvider, gpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, kCudaExecutionProvider);
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_GPU_Specific) {
  LayerAnnotation rule = {"gpu:nvidia", "Anno1", false};  // Assumes heuristics or vendor ID logic
  ExecutionProviders providers;

  // Add CPU provider
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add CUDA provider (NVIDIA vendor ID)
  auto gpu_ep = std::make_shared<MockExecutionProvider>(kCudaExecutionProvider,
                                                        OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0));
  ASSERT_STATUS_OK(providers.Add(kCudaExecutionProvider, gpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, kCudaExecutionProvider);
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_NoMatch) {
  LayerAnnotation rule = {"GPU", "Anno1", false};
  ExecutionProviders providers;

  // Only CPU provider available
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  EXPECT_FALSE(result.has_value());
}

TEST(EpLayeringMatcherTest, MatchExecutionProviders_Accelerator) {
  LayerAnnotation rule = {"accelerator", "Anno1", false};
  ExecutionProviders providers;

  // Add CPU
  auto cpu_ep = std::make_shared<MockExecutionProvider>(kCpuExecutionProvider, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add(kCpuExecutionProvider, cpu_ep));

  // Add custom accelerator
  auto accel_ep = std::make_shared<MockExecutionProvider>("MyAccel", OrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, 0, 0));
  ASSERT_STATUS_OK(providers.Add("MyAccel", accel_ep));

  auto result = EpLayeringMatcher::Match(providers, rule);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "MyAccel");
}

}  // namespace test
}  // namespace onnxruntime
