#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace teamredbench::native::mfu_gemm {

template <typename T>
struct KernelTuning;

template <typename T>
struct AccumulatorType {
    using type = T;
};

template <>
struct AccumulatorType<rocwmma::float16_t> {
    using type = float;
};

template <>
struct AccumulatorType<rocwmma::bfloat16_t> {
    using type = float;
};

template <typename T>
using AccumT = typename AccumulatorType<T>::type;

struct Args {
    std::string dtype = "float32";
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 10;
    int iterations = 50;
    int device_id = 0;
    int blocks_per_cu = 0;
};

[[noreturn]] inline void fail(const std::string& message) {
    throw std::runtime_error(message);
}

inline void check_hip(hipError_t status, const char* operation) {
    if(status != hipSuccess) {
        std::ostringstream buffer;
        buffer << operation << " failed: " << hipGetErrorString(status);
        fail(buffer.str());
    }
}

inline void ignore_hip(hipError_t status) {
    (void)status;
}

inline Args parse_args(int argc, char** argv) {
    Args args;
    for(int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if(index + 1 >= argc) {
            fail("missing value for argument " + key);
        }
        const std::string value = argv[++index];
        if(key == "--dtype") {
            args.dtype = value;
        } else if(key == "--m") {
            args.m = std::stoi(value);
        } else if(key == "--n") {
            args.n = std::stoi(value);
        } else if(key == "--k") {
            args.k = std::stoi(value);
        } else if(key == "--warmup") {
            args.warmup = std::stoi(value);
        } else if(key == "--iterations") {
            args.iterations = std::stoi(value);
        } else if(key == "--device-id") {
            args.device_id = std::stoi(value);
        } else if(key == "--blocks-per-cu") {
            args.blocks_per_cu = std::stoi(value);
        } else {
            fail("unknown argument " + key);
        }
    }
    if(args.m <= 0 || args.n <= 0 || args.k <= 0) {
        fail("matrix dimensions must be positive");
    }
    if(args.warmup < 0 || args.iterations < 0) {
        fail("warmup and iterations must be non-negative");
    }
    if(args.blocks_per_cu < 0) {
        fail("blocks-per-cu must be non-negative");
    }
    return args;
}

template <typename T>
__host__ __device__ inline T cast_scalar(double value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ inline rocwmma::bfloat16_t cast_scalar<rocwmma::bfloat16_t>(double value) {
    return rocwmma::bfloat16_t(static_cast<float>(value));
}

template <typename T, typename U>
__host__ __device__ inline T cast_output(U value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ inline rocwmma::bfloat16_t cast_output<rocwmma::bfloat16_t, float>(float value) {
    return rocwmma::bfloat16_t(value);
}

template <typename T>
constexpr int mfma_threads_per_block() {
    return 64 * KernelTuning<T>::kMfmaWaveGridM * KernelTuning<T>::kMfmaWaveGridN;
}

template <typename T>
constexpr int mfma_block_tile_m() {
    return KernelTuning<T>::kFragM * KernelTuning<T>::kMfmaWaveGridM * KernelTuning<T>::kMfmaWaveTileM;
}

template <typename T>
constexpr int mfma_block_tile_n() {
    return KernelTuning<T>::kFragN * KernelTuning<T>::kMfmaWaveGridN * KernelTuning<T>::kMfmaWaveTileN;
}

template <typename T>
constexpr int mfma_block_tile_k() {
    return KernelTuning<T>::kFragK * KernelTuning<T>::kMfmaKStages;
}

template <typename T>
inline int resolve_mfma_blocks_per_cu(const Args& args) {
    if(args.blocks_per_cu > 0) {
        return args.blocks_per_cu;
    }
    return KernelTuning<T>::kMfmaBlocksPerCu;
}

template <typename T>
__global__ void fill_kernel(T* data, std::size_t count, T value) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for(std::size_t index = tid; index < count; index += stride) {
        data[index] = value;
    }
}

inline int emit_skipped(const std::string& error) {
    std::cout << "{\"status\":\"skipped\",\"error\":\"";
    for(char character : error) {
        if(character == '"' || character == '\\') {
            std::cout << '\\';
        }
        if(character == '\n') {
            std::cout << ' ';
            continue;
        }
        std::cout << character;
    }
    std::cout << "\"}" << std::endl;
    return 0;
}

}  // namespace teamredbench::native::mfu_gemm
