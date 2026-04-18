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

constexpr int kThreadsPerBlock = 512;
constexpr int kComputeVectorUnroll = 16;
static_assert(kComputeVectorUnroll % 2 == 0, "double-buffered pipeline requires even unroll");
constexpr int kPipelineStage = kComputeVectorUnroll / 2;

using Vec16 = int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ Vec16 nt_load(const Vec16* __restrict__ p) {
    // Streaming loads: the working set (>=1 GiB) dwarfs L2, so bypassing the
    // cache with a nontemporal load frees L2 bandwidth for address/metadata
    // traffic and measurably lifts sustained HBM read BW on CDNA.
    return __builtin_nontemporal_load(p);
}

struct Args {
    std::string dtype = "float32";
    std::string mode = "copy";
    std::size_t size_mib = 1024;
    int warmup = 10;
    int iterations = 50;
    int device_id = 0;
    double scale = 1.0;
    int blocks_per_cu = 16;
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
        } else if(key == "--blocks-per-cu") {
            args.blocks_per_cu = std::stoi(value);
        } else {
            fail("unknown argument " + key);
        }
    }
    return args;
}

template <typename T>
__device__ __forceinline__ void scale_vector(Vec16& raw, T alpha) {
    constexpr std::size_t vec_width = sizeof(Vec16) / sizeof(T);
    T* lane = reinterpret_cast<T*>(&raw);
    #pragma unroll
    for(std::size_t k = 0; k < vec_width; ++k) {
        lane[k] = lane[k] * alpha;
    }
}

template <typename T>
__device__ __forceinline__ void triad_vector(Vec16& a, const Vec16& b, T alpha) {
    constexpr std::size_t vec_width = sizeof(Vec16) / sizeof(T);
    T* a_lane = reinterpret_cast<T*>(&a);
    const T* b_lane = reinterpret_cast<const T*>(&b);
    #pragma unroll
    for(std::size_t k = 0; k < vec_width; ++k) {
        a_lane[k] = a_lane[k] + b_lane[k] * alpha;
    }
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock) void scale_kernel(
    const T* __restrict__ src, T* __restrict__ dst, T alpha, std::size_t count) {
    constexpr std::size_t vec_width = sizeof(Vec16) / sizeof(T);

    const std::size_t vec_count = count / vec_width;
    const Vec16* __restrict__ src_v = reinterpret_cast<const Vec16*>(src);
    Vec16* __restrict__ dst_v = reinterpret_cast<Vec16*>(dst);

    // Block-chunk pattern: each block processes a contiguous kComputeVectorUnroll*blockDim
    // chunk of vectors so a wave's unrolled loads hit consecutive HBM rows (better row-
    // buffer locality than grid-stride with millions of elements between unrolled loads).
    const std::size_t block_chunk = static_cast<std::size_t>(blockDim.x) * kComputeVectorUnroll;
    const std::size_t grid_chunk = static_cast<std::size_t>(gridDim.x) * block_chunk;
    const std::size_t block_stride = blockDim.x;

    std::size_t base = static_cast<std::size_t>(blockIdx.x) * block_chunk + threadIdx.x;
    for(; base + (kComputeVectorUnroll - 1) * block_stride < vec_count; base += grid_chunk) {
        // Two-stage software pipeline: issue second-half loads while the first
        // half is still in flight/being computed, then overlap first-half
        // stores with second-half compute. Gives the scheduler strictly more
        // independent work across the load/compute/store phases.
        Vec16 r0[kPipelineStage];
        Vec16 r1[kPipelineStage];
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            r0[k] = nt_load(&src_v[base + k * block_stride]);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            r1[k] = nt_load(&src_v[base + (k + kPipelineStage) * block_stride]);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            scale_vector(r0[k], alpha);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            dst_v[base + k * block_stride] = r0[k];
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            scale_vector(r1[k], alpha);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            dst_v[base + (k + kPipelineStage) * block_stride] = r1[k];
        }
    }
    for(; base < vec_count; base += block_stride) {
        Vec16 raw = src_v[base];
        scale_vector(raw, alpha);
        dst_v[base] = raw;
    }

    const std::size_t tail_base = vec_count * vec_width;
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid_threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for(std::size_t j = tail_base + tid; j < count; j += grid_threads) {
        dst[j] = src[j] * alpha;
    }
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock) void triad_kernel(
    const T* __restrict__ src, const T* __restrict__ aux, T* __restrict__ dst,
    T alpha, std::size_t count) {
    constexpr std::size_t vec_width = sizeof(Vec16) / sizeof(T);

    const std::size_t vec_count = count / vec_width;
    const Vec16* __restrict__ src_v = reinterpret_cast<const Vec16*>(src);
    const Vec16* __restrict__ aux_v = reinterpret_cast<const Vec16*>(aux);
    Vec16* __restrict__ dst_v = reinterpret_cast<Vec16*>(dst);

    const std::size_t block_chunk = static_cast<std::size_t>(blockDim.x) * kComputeVectorUnroll;
    const std::size_t grid_chunk = static_cast<std::size_t>(gridDim.x) * block_chunk;
    const std::size_t block_stride = blockDim.x;

    std::size_t base = static_cast<std::size_t>(blockIdx.x) * block_chunk + threadIdx.x;
    for(; base + (kComputeVectorUnroll - 1) * block_stride < vec_count; base += grid_chunk) {
        // Two-stage pipeline with src/aux interleaved within each stage: the
        // HBM controller sees requests on both input buffers early, and the
        // second-half loads issue while the first half is computing/storing.
        Vec16 a0[kPipelineStage];
        Vec16 b0[kPipelineStage];
        Vec16 a1[kPipelineStage];
        Vec16 b1[kPipelineStage];
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            a0[k] = nt_load(&src_v[base + k * block_stride]);
            b0[k] = nt_load(&aux_v[base + k * block_stride]);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            a1[k] = nt_load(&src_v[base + (k + kPipelineStage) * block_stride]);
            b1[k] = nt_load(&aux_v[base + (k + kPipelineStage) * block_stride]);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            triad_vector(a0[k], b0[k], alpha);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            dst_v[base + k * block_stride] = a0[k];
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            triad_vector(a1[k], b1[k], alpha);
        }
        #pragma unroll
        for(int k = 0; k < kPipelineStage; ++k) {
            dst_v[base + (k + kPipelineStage) * block_stride] = a1[k];
        }
    }
    for(; base < vec_count; base += block_stride) {
        Vec16 a = src_v[base];
        const Vec16 b = aux_v[base];
        triad_vector(a, b, alpha);
        dst_v[base] = a;
    }

    const std::size_t tail_base = vec_count * vec_width;
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t grid_threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for(std::size_t j = tail_base + tid; j < count; j += grid_threads) {
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
    const int blocks_per_cu = std::max(1, args.blocks_per_cu);
    const int max_blocks = std::max(1, props.multiProcessorCount * blocks_per_cu);
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
