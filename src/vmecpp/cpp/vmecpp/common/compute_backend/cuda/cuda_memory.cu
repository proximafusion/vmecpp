// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/cuda/cuda_memory.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace vmecpp {

namespace {

void CheckCudaError(cudaError_t err, const char* operation) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + operation + ": " +
                             cudaGetErrorString(err));
  }
}

}  // namespace

// PinnedBuffer implementation

PinnedBuffer::PinnedBuffer() : data_(nullptr), size_(0) {}

PinnedBuffer::PinnedBuffer(size_t size_bytes) : data_(nullptr), size_(0) {
  Resize(size_bytes);
}

PinnedBuffer::~PinnedBuffer() {
  if (data_ != nullptr) {
    cudaFreeHost(data_);
  }
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
    }
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void PinnedBuffer::Resize(size_t new_size_bytes) {
  if (new_size_bytes > size_) {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
    }
    CheckCudaError(cudaMallocHost(&data_, new_size_bytes), "cudaMallocHost");
    size_ = new_size_bytes;
  }
}

// DeviceBuffer implementation

DeviceBuffer::DeviceBuffer() : data_(nullptr), size_(0) {}

DeviceBuffer::DeviceBuffer(size_t size_bytes) : data_(nullptr), size_(0) {
  Resize(size_bytes);
}

DeviceBuffer::~DeviceBuffer() {
  if (data_ != nullptr) {
    cudaFree(data_);
  }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void DeviceBuffer::Resize(size_t new_size_bytes) {
  if (new_size_bytes > size_) {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
    }
    CheckCudaError(cudaMalloc(&data_, new_size_bytes), "cudaMalloc");
    size_ = new_size_bytes;
  }
}

void DeviceBuffer::SetZero(cudaStream_t stream) {
  if (data_ != nullptr && size_ > 0) {
    if (stream != nullptr) {
      CheckCudaError(cudaMemsetAsync(data_, 0, size_, stream),
                     "cudaMemsetAsync");
    } else {
      CheckCudaError(cudaMemset(data_, 0, size_), "cudaMemset");
    }
  }
}

void DeviceBuffer::CopyFromHost(const void* src, size_t size_bytes,
                                cudaStream_t stream) {
  Resize(size_bytes);
  if (stream != nullptr) {
    CheckCudaError(
        cudaMemcpyAsync(data_, src, size_bytes, cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync H2D");
  } else {
    CheckCudaError(cudaMemcpy(data_, src, size_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
  }
}

void DeviceBuffer::CopyToHost(void* dst, size_t size_bytes,
                              cudaStream_t stream) const {
  if (stream != nullptr) {
    CheckCudaError(
        cudaMemcpyAsync(dst, data_, size_bytes, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync D2H");
  } else {
    CheckCudaError(cudaMemcpy(dst, data_, size_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
  }
}

void DeviceBuffer::CopyFromPinned(const PinnedBuffer& src, size_t size_bytes,
                                  cudaStream_t stream) {
  Resize(size_bytes);
  CheckCudaError(cudaMemcpyAsync(data_, src.Data(), size_bytes,
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync H2D (pinned)");
}

void DeviceBuffer::CopyToPinned(PinnedBuffer& dst, size_t size_bytes,
                                cudaStream_t stream) const {
  dst.Resize(size_bytes);
  CheckCudaError(cudaMemcpyAsync(dst.Data(), data_, size_bytes,
                                 cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync D2H (pinned)");
}

// CudaStream implementation

CudaStream::CudaStream() {
  CheckCudaError(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

void CudaStream::Synchronize() const {
  if (stream_ != nullptr) {
    CheckCudaError(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }
}

bool CudaStream::IsComplete() const {
  if (stream_ == nullptr) {
    return true;
  }
  cudaError_t status = cudaStreamQuery(stream_);
  if (status == cudaSuccess) {
    return true;
  } else if (status == cudaErrorNotReady) {
    return false;
  } else {
    CheckCudaError(status, "cudaStreamQuery");
    return false;  // Never reached
  }
}

// StreamManager implementation

StreamManager::StreamManager(int num_streams) {
  if (num_streams <= 0) {
    throw std::invalid_argument("StreamManager: num_streams must be positive");
  }
  streams_.reserve(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    streams_.emplace_back();
  }
}

cudaStream_t StreamManager::GetStream(int index) const {
  return streams_[index % streams_.size()].Get();
}

void StreamManager::SynchronizeAll() const {
  for (const auto& stream : streams_) {
    stream.Synchronize();
  }
}

// TileTransferManager implementation

TileTransferManager::TileTransferManager(size_t buffer_size_bytes,
                                         int num_buffers)
    : buffer_size_(buffer_size_bytes) {
  if (num_buffers <= 0) {
    throw std::invalid_argument(
        "TileTransferManager: num_buffers must be positive");
  }

  input_buffers_.reserve(num_buffers);
  output_buffers_.reserve(num_buffers);
  device_buffers_.reserve(num_buffers);

  for (int i = 0; i < num_buffers; ++i) {
    input_buffers_.emplace_back(buffer_size_bytes);
    output_buffers_.emplace_back(buffer_size_bytes);
    device_buffers_.emplace_back(buffer_size_bytes);
  }
}

PinnedBuffer& TileTransferManager::GetInputBuffer(int buffer_idx) {
  return input_buffers_[buffer_idx % input_buffers_.size()];
}

PinnedBuffer& TileTransferManager::GetOutputBuffer(int buffer_idx) {
  return output_buffers_[buffer_idx % output_buffers_.size()];
}

DeviceBuffer& TileTransferManager::GetDeviceBuffer(int buffer_idx) {
  return device_buffers_[buffer_idx % device_buffers_.size()];
}

void TileTransferManager::StageInput(const void* src, size_t size_bytes,
                                     int buffer_idx) {
  auto& buffer = GetInputBuffer(buffer_idx);
  buffer.Resize(size_bytes);
  std::memcpy(buffer.Data(), src, size_bytes);
}

void TileTransferManager::UnstageOutput(void* dst, size_t size_bytes,
                                        int buffer_idx) {
  const auto& buffer = GetOutputBuffer(buffer_idx);
  std::memcpy(dst, buffer.Data(), size_bytes);
}

void TileTransferManager::TransferToDevice(int buffer_idx,
                                           cudaStream_t stream) {
  auto& pinned = GetInputBuffer(buffer_idx);
  auto& device = GetDeviceBuffer(buffer_idx);
  device.CopyFromPinned(pinned, pinned.Size(), stream);
}

void TileTransferManager::TransferFromDevice(int buffer_idx, size_t size_bytes,
                                             cudaStream_t stream) {
  const auto& device = GetDeviceBuffer(buffer_idx);
  auto& pinned = GetOutputBuffer(buffer_idx);
  device.CopyToPinned(pinned, size_bytes, stream);
}

}  // namespace vmecpp
