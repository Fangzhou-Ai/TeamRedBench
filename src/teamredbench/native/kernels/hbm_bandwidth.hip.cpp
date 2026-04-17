#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kCopyVectorUnroll = 4;
constexpr int kComputeVectorUnroll = 2;

struct Args {
    std::string dtype = "float32";
    std::string mode = "copy";
    std::size_t size_mib = 1024;
    int warmup = 10;
    int iterations = 50;
    int device_id = 0;
    double scale = 1.0;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check_hip(hipError_t status, const char* operation) {
    if(status != hipSuccess) {
        std::ostringstream buffer;
        buffer << operation << " failed: " << hipGetErrorString(status);
        fail(buffer.str());
    }
}

Args parse_args(int argc, char** argv) {
    Args args;
    for(int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if(index + 1 >= argc) {
            fail("missing value for argument " + key);
        }
        const std::string value = argv[++index];
        if(key == "--dtype") {
            args.dtype = value;
        } else if(key == "--mode") {
            args.mode = value;
        } else if(key == "--size-mib") {
            args.size_mib = static_cast<std::size_t>(std::stoull(value));
        } else if(key == "--warmup") {
            args.warmup = std::stoi(value);
        } else if(key == "--iterations") {
            args.iterations = std::stoi(value);
        } else if(key == "--device-id") {
            args.device_id = std::stoi(value);
        } else if(key == "--scale") {
            args.scale = std::stod(value);
        } else {
            fail("unknown argument " + key);
        }
    }
    return args;
}

template <typename T>
__device__ __forceinline__ void scale_vector(int4& raw, T alpha) {
    constexpr std::size_t vec_width = sizeof(int4) / sizeof(T);
    T* lane = reinterpret_cast<T*>(&raw);
    #pragma unroll
    for(std::size_t k = 0; k < vec_width; ++k) {
        lane[k] = lane[k] * alpha;
    }
}

template <typename T>
__device__ __forceinline__ void triad_vector(int4& a, const int4& b, T alpha) {
    constexpr std::size_t vec_width = sizeof(int4) / sizeof(T);
    T* a_lane = reinterpret_cast<T*>(&a);
    const T* b_lane = reinterpret_cast<const T*>(&b);
    #pragma unroll
    for(std::size_t k = 0; k < vec_width; ++k) {
        a_lane[k] = a_lane[k] + b_lane[k] * alpha;
    }
}

template <typename T>
__global__ __launch_bounds__(256) void scale_kernel(
    const T* __restrict__ src, T* __restrict__ dst, T alpha, std::size_t count) {
    using Vec = int4;
    constexpr std::size_t vec_width = sizeof(Vec) / sizeof(T);

    const std::size_t vec_count = count / vec_width;
    const Vec* __restrict__ src_v = reinterpret_cast<const Vec*>(src);
    Vec* __restrict__ dst_v = reinterpret_cast<Vec*>(dst);

    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    std::size_t index = tid;
    const std::size_t unroll_bound =
        vec_count >= static_cast<std::size_t>(kComputeVectorUnroll - 1) * stride
            ? vec_count - static_cast<std::size_t>(kComputeVectorUnroll - 1) * stride
            : 0;
    for(; index < unroll_bound; index += static_cast<std::size_t>(kComputeVectorUnroll) * stride) {
        Vec raw0 = src_v[index];
        Vec raw1 = src_v[index + stride];
        scale_vector(raw0, alpha);
        scale_vector(raw1, alpha);
        dst_v[index] = raw0;
        dst_v[index + stride] = raw1;
    }
    for(; index < vec_count; index += stride) {
        Vec raw = src_v[index];
        scale_vector(raw, alpha);
        dst_v[index] = raw;
    }

    const std::size_t tail_base = vec_count * vec_width;
    for(std::size_t j = tail_base + tid; j < count; j += stride) {
        dst[j] = src[j] * alpha;
    }
}

template <typename T>
__global__ __launch_bounds__(256) void triad_kernel(
    const T* __restrict__ src, const T* __restrict__ aux, T* __restrict__ dst,
    T alpha, std::size_t count) {
    using Vec = int4;
    constexpr std::size_t vec_width = sizeof(Vec) / sizeof(T);

    const std::size_t vec_count = count / vec_width;
    const Vec* __restrict__ src_v = reinterpret_cast<const Vec*>(src);
    const Vec* __restrict__ aux_v = reinterpret_cast<const Vec*>(aux);
    Vec* __restrict__ dst_v = reinterpret_cast<Vec*>(dst);

    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    std::size_t index = tid;
    const std::size_t unroll_bound =
        vec_count >= static_cast<std::size_t>(kComputeVectorUnroll - 1) * stride
            ? vec_count - static_cast<std::size_t>(kComputeVectorUnroll - 1) * stride
            : 0;
    for(; index < unroll_bound; index += static_cast<std::size_t>(kComputeVectorUnroll) * stride) {
        Vec a0 = src_v[index];
        Vec b0 = aux_v[index];
        Vec a1 = src_v[index + stride];
        Vec b1 = aux_v[index + stride];
        triad_vector(a0, b0, alpha);
        triad_vector(a1, b1, alpha);
        dst_v[index] = a0;
        dst_v[index + stride] = a1;
    }
    for(; index < vec_count; index += stride) {
        Vec a = src_v[index];
        const Vec b = aux_v[index];
        triad_vector(a, b, alpha);
        dst_v[index] = a;
    }

    const std::size_t tail_base = vec_count * vec_width;
    for(std::size_t j = tail_base + tid; j < count; j += stride) {
        dst[j] = src[j] + aux[j] * alpha;
    }
}

template <typename T>
double run_kernel(const Args& args) {
    check_hip(hipSetDevice(args.device_id), "hipSetDevice");

    const std::size_t total_bytes = args.size_mib * 1024ULL * 1024ULL;
    const std::size_t count = std::max<std::size_t>(1, total_bytes / sizeof(T));
    const std::size_t data_bytes = count * sizeof(T);

    T* src = nullptr;
    T* dst = nullptr;
    T* aux = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, args.device_id), "hipGetDeviceProperties");

    check_hip(hipMalloc(&reinterpret_cast<void*&>(src), data_bytes), "hipMalloc(src)");
    check_hip(hipMalloc(&reinterpret_cast<void*&>(dst), data_bytes), "hipMalloc(dst)");
    if(args.mode == "triad") {
        check_hip(hipMalloc(&reinterpret_cast<void*&>(aux), data_bytes), "hipMalloc(aux)");
    }

    check_hip(hipMemset(src, 0, data_bytes), "hipMemset(src)");
    check_hip(hipMemset(dst, 0, data_bytes), "hipMemset(dst)");
    if(aux != nullptr) {
        check_hip(hipMemset(aux, 0, data_bytes), "hipMemset(aux)");
    }

    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");

    const int threads = kThreadsPerBlock;
    // A wide grid keeps multiple wavefronts per CU queued without pushing the
    // per-thread work chunk so small that loop overhead dominates.
    constexpr std::size_t vec_bytes = 16;
    const std::size_t vec_count = data_bytes / vec_bytes;
    const int max_blocks = std::max(1, props.multiProcessorCount * 16);
    const int blocks_by_work = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(max_blocks),
        std::max<std::size_t>(1, (vec_count + static_cast<std::size_t>(threads) - 1)
                                 / static_cast<std::size_t>(threads))
    ));
    const int blocks = std::max(1, blocks_by_work);
    const T alpha = static_cast<T>(args.scale);

    auto launch_once = [&]() {
        if(args.mode == "copy") {
            check_hip(hipMemcpyAsync(dst, src, data_bytes, hipMemcpyDeviceToDevice, stream), "hipMemcpyAsync");
            return;
        }
        if(args.mode == "scale") {
            hipLaunchKernelGGL(scale_kernel<T>, dim3(blocks), dim3(threads), 0, stream, src, dst, alpha, count);
            check_hip(hipGetLastError(), "scale_kernel");
            return;
        }
        if(args.mode == "triad") {
            hipLaunchKernelGGL(triad_kernel<T>, dim3(blocks), dim3(threads), 0, stream, src, aux, dst, alpha, count);
            check_hip(hipGetLastError(), "triad_kernel");
            return;
        }
        fail("unsupported mode " + args.mode);
    };

    for(int iteration = 0; iteration < args.warmup; ++iteration) {
        launch_once();
    }
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(warmup)");

    check_hip(hipEventRecord(start, stream), "hipEventRecord(start)");
    for(int iteration = 0; iteration < args.iterations; ++iteration) {
        launch_once();
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");

    if(stop != nullptr) {
        hipEventDestroy(stop);
    }
    if(start != nullptr) {
        hipEventDestroy(start);
    }
    if(stream != nullptr) {
        hipStreamDestroy(stream);
    }
    if(aux != nullptr) {
        hipFree(aux);
    }
    if(dst != nullptr) {
        hipFree(dst);
    }
    if(src != nullptr) {
        hipFree(src);
    }

    const int iters = std::max(args.iterations, 1);
    return static_cast<double>(elapsed_ms) / 1000.0 / static_cast<double>(iters);
}

template <typename T>
int emit_success(const Args& args) {
    const double elapsed_s = run_kernel<T>(args);
    std::cout << "{\"status\":\"ok\",\"raw_metrics\":{\"elapsed_s\":" << elapsed_s
              << "},\"metadata\":{\"implementation\":\"hip\"}}" << std::endl;
    return 0;
}

int emit_skipped(const std::string& error) {
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if(args.dtype == "float16") {
            return emit_success<__half>(args);
        }
        if(args.dtype == "bfloat16") {
            return emit_success<hip_bfloat16>(args);
        }
        if(args.dtype == "float32") {
            return emit_success<float>(args);
        }
        if(args.dtype == "float64") {
            return emit_success<double>(args);
        }
        if(args.dtype == "int8" && args.mode == "copy") {
            return emit_success<std::int8_t>(args);
        }
        return emit_skipped("unsupported dtype or mode for native HBM kernel: " + args.dtype + "/" + args.mode);
    } catch(const std::exception& exc) {
        return emit_skipped(exc.what());
    }
}
