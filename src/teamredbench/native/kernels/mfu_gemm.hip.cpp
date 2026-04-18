#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

template <typename T>
struct KernelTuning {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kBlockTileK = 8;
    static constexpr int kBlocksPerCu = 8;
};

template <>
struct KernelTuning<__half> {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kBlockTileK = 16;
    static constexpr int kBlocksPerCu = 8;
};

template <>
struct KernelTuning<hip_bfloat16> {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kBlockTileK = 16;
    static constexpr int kBlocksPerCu = 6;
};

template <>
struct KernelTuning<double> {
    static constexpr int kThreadsX = 8;
    static constexpr int kThreadsY = 8;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kBlockTileK = 4;
    static constexpr int kBlocksPerCu = 16;
};

template <typename T>
struct AccumulatorType {
    using type = T;
};

template <>
struct AccumulatorType<__half> {
    using type = float;
};

template <>
struct AccumulatorType<hip_bfloat16> {
    using type = float;
};

template <typename T>
using AccumT = typename AccumulatorType<T>::type;

template <typename T>
constexpr int threads_per_block() {
    return KernelTuning<T>::kThreadsX * KernelTuning<T>::kThreadsY;
}

template <typename T>
constexpr int block_tile_m() {
    return KernelTuning<T>::kThreadsY * KernelTuning<T>::kThreadTileM;
}

template <typename T>
constexpr int block_tile_n() {
    return KernelTuning<T>::kThreadsX * KernelTuning<T>::kThreadTileN;
}

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

void ignore_hip(hipError_t status) {
    (void)status;
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
__host__ __device__ T cast_scalar(double value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ __half cast_scalar<__half>(double value) {
    return __half(static_cast<float>(value));
}

template <>
__host__ __device__ hip_bfloat16 cast_scalar<hip_bfloat16>(double value) {
    return hip_bfloat16(static_cast<float>(value));
}

template <typename T>
__host__ __device__ AccumT<T> to_accum(T value) {
    return static_cast<AccumT<T>>(value);
}

template <typename T>
int resolve_blocks_per_cu(const Args& args) {
    if(args.blocks_per_cu > 0) {
        return args.blocks_per_cu;
    }
    return KernelTuning<T>::kBlocksPerCu;
}

template <typename T>
__global__ void fill_kernel(T* data, std::size_t count, T value) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for(std::size_t index = tid; index < count; index += stride) {
        data[index] = value;
    }
}

template <
    typename T,
    int ThreadsX = KernelTuning<T>::kThreadsX,
    int ThreadsY = KernelTuning<T>::kThreadsY,
    int ThreadTileM = KernelTuning<T>::kThreadTileM,
    int ThreadTileN = KernelTuning<T>::kThreadTileN,
    int BlockTileK = KernelTuning<T>::kBlockTileK>
__global__ __launch_bounds__(ThreadsX * ThreadsY) void gemm_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int m,
    int n,
    int k) {
    using Acc = AccumT<T>;
    constexpr int kThreadsPerBlock = ThreadsX * ThreadsY;
    constexpr int kBlockTileM = ThreadsY * ThreadTileM;
    constexpr int kBlockTileN = ThreadsX * ThreadTileN;

    static_assert(kBlockTileM > 0 && kBlockTileN > 0 && BlockTileK > 0, "tile sizes must be positive");
    static_assert(kThreadsPerBlock > 0, "threads per block must be positive");

    __shared__ T a_tile[kBlockTileM * BlockTileK];
    __shared__ T b_tile[BlockTileK * kBlockTileN];

    const int linear_tid = static_cast<int>(threadIdx.y) * ThreadsX + static_cast<int>(threadIdx.x);
    const int tiles_n = (n + kBlockTileN - 1) / kBlockTileN;
    const int tiles_m = (m + kBlockTileM - 1) / kBlockTileM;
    const int total_tiles = tiles_m * tiles_n;

    for(int tile_linear = static_cast<int>(blockIdx.x); tile_linear < total_tiles; tile_linear += static_cast<int>(gridDim.x)) {
        const int tile_row = tile_linear / tiles_n;
        const int tile_col = tile_linear % tiles_n;
        const int row_base = tile_row * kBlockTileM;
        const int col_base = tile_col * kBlockTileN;
        const int thread_row = static_cast<int>(threadIdx.y) * ThreadTileM;
        const int thread_col = static_cast<int>(threadIdx.x) * ThreadTileN;

        Acc accum[ThreadTileM][ThreadTileN];
        #pragma unroll
        for(int i = 0; i < ThreadTileM; ++i) {
            #pragma unroll
            for(int j = 0; j < ThreadTileN; ++j) {
                accum[i][j] = static_cast<Acc>(0);
            }
        }

        for(int k_base = 0; k_base < k; k_base += BlockTileK) {
            for(int index = linear_tid; index < kBlockTileM * BlockTileK; index += kThreadsPerBlock) {
                const int tile_row_offset = index / BlockTileK;
                const int tile_k_offset = index % BlockTileK;
                const int global_row = row_base + tile_row_offset;
                const int global_k = k_base + tile_k_offset;
                a_tile[index] = (global_row < m && global_k < k)
                    ? a[static_cast<std::size_t>(global_row) * static_cast<std::size_t>(k) + static_cast<std::size_t>(global_k)]
                    : cast_scalar<T>(0.0);
            }

            for(int index = linear_tid; index < BlockTileK * kBlockTileN; index += kThreadsPerBlock) {
                const int tile_k_offset = index / kBlockTileN;
                const int tile_col_offset = index % kBlockTileN;
                const int global_k = k_base + tile_k_offset;
                const int global_col = col_base + tile_col_offset;
                b_tile[index] = (global_k < k && global_col < n)
                    ? b[static_cast<std::size_t>(global_k) * static_cast<std::size_t>(n) + static_cast<std::size_t>(global_col)]
                    : cast_scalar<T>(0.0);
            }
            __syncthreads();

            #pragma unroll
            for(int kk = 0; kk < BlockTileK; ++kk) {
                Acc a_frag[ThreadTileM];
                Acc b_frag[ThreadTileN];

                #pragma unroll
                for(int i = 0; i < ThreadTileM; ++i) {
                    a_frag[i] = to_accum(a_tile[(thread_row + i) * BlockTileK + kk]);
                }
                #pragma unroll
                for(int j = 0; j < ThreadTileN; ++j) {
                    b_frag[j] = to_accum(b_tile[kk * kBlockTileN + thread_col + j]);
                }

                #pragma unroll
                for(int i = 0; i < ThreadTileM; ++i) {
                    #pragma unroll
                    for(int j = 0; j < ThreadTileN; ++j) {
                        accum[i][j] += a_frag[i] * b_frag[j];
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for(int i = 0; i < ThreadTileM; ++i) {
            const int global_row = row_base + thread_row + i;
            if(global_row >= m) {
                continue;
            }
            #pragma unroll
            for(int j = 0; j < ThreadTileN; ++j) {
                const int global_col = col_base + thread_col + j;
                if(global_col >= n) {
                    continue;
                }
                c[static_cast<std::size_t>(global_row) * static_cast<std::size_t>(n) + static_cast<std::size_t>(global_col)] =
                    cast_scalar<T>(static_cast<double>(accum[i][j]));
            }
        }
    }
}

template <typename T>
double run_kernel(const Args& args) {
    check_hip(hipSetDevice(args.device_id), "hipSetDevice");

    const std::size_t a_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.k);
    const std::size_t b_count = static_cast<std::size_t>(args.k) * static_cast<std::size_t>(args.n);
    const std::size_t c_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.n);

    T* a = nullptr;
    T* b = nullptr;
    T* c = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, args.device_id), "hipGetDeviceProperties");

    check_hip(hipMalloc(&reinterpret_cast<void*&>(a), a_count * sizeof(T)), "hipMalloc(a)");
    check_hip(hipMalloc(&reinterpret_cast<void*&>(b), b_count * sizeof(T)), "hipMalloc(b)");
    check_hip(hipMalloc(&reinterpret_cast<void*&>(c), c_count * sizeof(T)), "hipMalloc(c)");

    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");

    constexpr int kInitThreads = 256;
    const std::size_t max_count = std::max(a_count, std::max(b_count, c_count));
    const int init_blocks = std::max(
        1,
        static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(std::max(1, props.multiProcessorCount * 8)),
            (max_count + static_cast<std::size_t>(kInitThreads) - 1) / static_cast<std::size_t>(kInitThreads)
        ))
    );

    hipLaunchKernelGGL(fill_kernel<T>, dim3(init_blocks), dim3(kInitThreads), 0, stream, a, a_count, cast_scalar<T>(0.03125));
    check_hip(hipGetLastError(), "fill_kernel(a)");
    hipLaunchKernelGGL(fill_kernel<T>, dim3(init_blocks), dim3(kInitThreads), 0, stream, b, b_count, cast_scalar<T>(0.03125));
    check_hip(hipGetLastError(), "fill_kernel(b)");
    hipLaunchKernelGGL(fill_kernel<T>, dim3(init_blocks), dim3(kInitThreads), 0, stream, c, c_count, cast_scalar<T>(0.0));
    check_hip(hipGetLastError(), "fill_kernel(c)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(init)");

    constexpr int kThreadsX = KernelTuning<T>::kThreadsX;
    constexpr int kThreadsY = KernelTuning<T>::kThreadsY;
    constexpr int kBlockTileM = block_tile_m<T>();
    constexpr int kBlockTileN = block_tile_n<T>();
    const int tiles_m = (args.m + kBlockTileM - 1) / kBlockTileM;
    const int tiles_n = (args.n + kBlockTileN - 1) / kBlockTileN;
    const int total_tiles = std::max(1, tiles_m * tiles_n);
    const int blocks = std::max(1, std::min(total_tiles, props.multiProcessorCount * std::max(1, resolve_blocks_per_cu<T>(args))));
    const dim3 threads(kThreadsX, kThreadsY);

    auto launch_once = [&]() {
        hipLaunchKernelGGL(gemm_kernel<T>, dim3(blocks), threads, 0, stream, a, b, c, args.m, args.n, args.k);
        check_hip(hipGetLastError(), "gemm_kernel");
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
        ignore_hip(hipEventDestroy(stop));
    }
    if(start != nullptr) {
        ignore_hip(hipEventDestroy(start));
    }
    if(stream != nullptr) {
        ignore_hip(hipStreamDestroy(stream));
    }
    if(c != nullptr) {
        ignore_hip(hipFree(c));
    }
    if(b != nullptr) {
        ignore_hip(hipFree(b));
    }
    if(a != nullptr) {
        ignore_hip(hipFree(a));
    }

    const int iters = std::max(args.iterations, 1);
    return static_cast<double>(elapsed_ms) / 1000.0 / static_cast<double>(iters);
}

template <typename T>
int emit_success(const Args& args) {
    const double elapsed_s = run_kernel<T>(args);
    std::cout << "{\"status\":\"ok\",\"raw_metrics\":{\"elapsed_s\":" << elapsed_s
              << "},\"metadata\":{\"implementation\":\"hip\""
              << ",\"threads_per_block\":" << threads_per_block<T>()
              << ",\"thread_tile_m\":" << KernelTuning<T>::kThreadTileM
              << ",\"thread_tile_n\":" << KernelTuning<T>::kThreadTileN
              << ",\"tile_m\":" << block_tile_m<T>()
              << ",\"tile_n\":" << block_tile_n<T>()
              << ",\"tile_k\":" << KernelTuning<T>::kBlockTileK
              << ",\"blocks_per_cu\":" << resolve_blocks_per_cu<T>(args)
              << "}}" << std::endl;
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
        return emit_skipped("unsupported dtype for native MFU kernel: " + args.dtype);
    } catch(const std::exception& exc) {
        return emit_skipped(exc.what());
    }
}
