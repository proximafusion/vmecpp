// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#ifndef VMECPP_COMMON_COMPUTE_BACKEND_CUDA_CUDA_MEMORY_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_CUDA_CUDA_MEMORY_H_

#include <cstddef>
#include <memory>
#include <vector>

// Forward declarations for CUDA types
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace vmecpp {

// RAII wrapper for pinned (page-locked) host memory.
// Pinned memory enables async DMA transfers between host and device.
class PinnedBuffer {
 public:
  PinnedBuffer();
  explicit PinnedBuffer(size_t size_bytes);
  ~PinnedBuffer();

  // Non-copyable
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  // Movable
  PinnedBuffer(PinnedBuffer&& other) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

  // Resize buffer (reallocates if larger)
  void Resize(size_t new_size_bytes);

  // Access raw pointer
  void* Data() { return data_; }
  const void* Data() const { return data_; }

  // Typed access
  template <typename T>
  T* DataAs() {
    return static_cast<T*>(data_);
  }
  template <typename T>
  const T* DataAs() const {
    return static_cast<const T*>(data_);
  }

  // Size in bytes
  size_t Size() const { return size_; }

  // Check if allocated
  bool IsAllocated() const { return data_ != nullptr; }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

// RAII wrapper for CUDA device memory buffer.
class DeviceBuffer {
 public:
  DeviceBuffer();
  explicit DeviceBuffer(size_t size_bytes);
  ~DeviceBuffer();

  // Non-copyable
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Movable
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  // Resize buffer (reallocates if larger)
  void Resize(size_t new_size_bytes);

  // Access raw pointer
  void* Data() { return data_; }
  const void* Data() const { return data_; }

  // Typed access
  template <typename T>
  T* DataAs() {
    return static_cast<T*>(data_);
  }
  template <typename T>
  const T* DataAs() const {
    return static_cast<const T*>(data_);
  }

  // Size in bytes
  size_t Size() const { return size_; }

  // Check if allocated
  bool IsAllocated() const { return data_ != nullptr; }

  // Set memory to zero
  void SetZero(cudaStream_t stream = nullptr);

  // Copy from host memory
  void CopyFromHost(const void* src, size_t size_bytes,
                    cudaStream_t stream = nullptr);

  // Copy to host memory
  void CopyToHost(void* dst, size_t size_bytes,
                  cudaStream_t stream = nullptr) const;

  // Copy from pinned buffer (async capable)
  void CopyFromPinned(const PinnedBuffer& src, size_t size_bytes,
                      cudaStream_t stream);

  // Copy to pinned buffer (async capable)
  void CopyToPinned(PinnedBuffer& dst, size_t size_bytes,
                    cudaStream_t stream) const;

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

// RAII wrapper for CUDA stream.
class CudaStream {
 public:
  CudaStream();
  ~CudaStream();

  // Non-copyable
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  // Movable
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;

  // Get raw stream handle
  cudaStream_t Get() const { return stream_; }

  // Synchronize stream
  void Synchronize() const;

  // Check if stream has completed all operations
  bool IsComplete() const;

 private:
  cudaStream_t stream_ = nullptr;
};

// Manages multiple CUDA streams for pipelined execution.
class StreamManager {
 public:
  explicit StreamManager(int num_streams);
  ~StreamManager() = default;

  // Get stream by index (wraps around)
  cudaStream_t GetStream(int index) const;

  // Synchronize all streams
  void SynchronizeAll() const;

  // Number of streams
  int NumStreams() const { return static_cast<int>(streams_.size()); }

 private:
  std::vector<CudaStream> streams_;
};

// Double-buffered tile transfer manager for async pipelining.
class TileTransferManager {
 public:
  // Create manager with given buffer size and number of buffers
  TileTransferManager(size_t buffer_size_bytes, int num_buffers = 2);

  // Get input staging buffer for given buffer index
  PinnedBuffer& GetInputBuffer(int buffer_idx);

  // Get output staging buffer for given buffer index
  PinnedBuffer& GetOutputBuffer(int buffer_idx);

  // Get device buffer for given buffer index
  DeviceBuffer& GetDeviceBuffer(int buffer_idx);

  // Stage data from user memory to pinned input buffer
  void StageInput(const void* src, size_t size_bytes, int buffer_idx);

  // Unstage data from pinned output buffer to user memory
  void UnstageOutput(void* dst, size_t size_bytes, int buffer_idx);

  // Transfer input buffer to device (async)
  void TransferToDevice(int buffer_idx, cudaStream_t stream);

  // Transfer device buffer to output (async)
  void TransferFromDevice(int buffer_idx, size_t size_bytes,
                          cudaStream_t stream);

  // Number of buffers
  int NumBuffers() const { return static_cast<int>(input_buffers_.size()); }

 private:
  std::vector<PinnedBuffer> input_buffers_;
  std::vector<PinnedBuffer> output_buffers_;
  std::vector<DeviceBuffer> device_buffers_;
  size_t buffer_size_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_CUDA_CUDA_MEMORY_H_
