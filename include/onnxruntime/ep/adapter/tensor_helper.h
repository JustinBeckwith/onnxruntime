// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/_pch.h instead."
#endif

#include <memory>
#include <sstream>

#include "core/framework/tensor.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// Create an unowned onnxruntime::Tensor from a tensor OrtValue from C API.
/// </summary>
inline std::unique_ptr<onnxruntime::Tensor> CreateTensorFromApiValue(const OrtValue* ort_value) {
  Ort::ConstValue value{ort_value};
  EP_ENFORCE(value.IsTensor(), "Only tensor OrtValue is supported.");

  auto type_and_shape_info = value.GetTypeInfo().GetTensorTypeAndShapeInfo();
  auto type = type_and_shape_info.GetElementType();
  auto shape_vec = type_and_shape_info.GetShape();

  auto memory_info = value.GetTensorMemoryInfo();
  MLDataType data_type = DataTypeImpl::TensorTypeFromONNXEnum(type)->GetElementType();

  return std::make_unique<Tensor>(data_type,
                                  TensorShape{shape_vec},
                                  const_cast<void*>(value.GetTensorRawData()),
                                  OrtMemoryInfo{
                                      memory_info.GetAllocatorName(),
                                      memory_info.GetAllocatorType(),
                                      OrtDevice{
                                          static_cast<OrtDevice::DeviceType>(memory_info.GetDeviceType()),
                                          static_cast<OrtDevice::MemoryType>(memory_info.GetMemoryType()),
                                          static_cast<OrtDevice::VendorId>(memory_info.GetVendorId()),
                                          static_cast<OrtDevice::DeviceId>(memory_info.GetDeviceId()),

                                      },
                                      memory_info.GetMemoryType()});
}

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
