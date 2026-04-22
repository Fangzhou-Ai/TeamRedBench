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

namespace {

template <typename T>
struct KernelTuning {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kScalarBlockTileK = 8;
    static constexpr int kScalarBlocksPerCu = 8;

    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 16;
    static constexpr int kMfmaWaveGridM = 2;
    static constexpr int kMfmaWaveGridN = 2;
    static constexpr int kMfmaWaveTileM = 2;
    static constexpr int kMfmaWaveTileN = 2;
    static constexpr int kMfmaKStages = 2;
    static constexpr int kMfmaBlocksPerCu = 4;
    static constexpr int kMfmaTileGroupM = 8;
    static constexpr int kMfmaLdsPadA = 0;
    static constexpr int kMfmaLdsPadB = 0;
};

template <>
struct KernelTuning<rocwmma::float16_t> {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kScalarBlockTileK = 16;
    static constexpr int kScalarBlocksPerCu = 8;

    // gfx950 exposes the 16x16x32 f16 MFMA. A 256x128 macro tile keeps strong
    // data reuse; using 8 waves per block restores 16 waves/CU on the 4096^3
    // smoke case while still fitting two resident blocks. Skew B rows by one
    // 16-byte vector to break the worst LDS stride-aligned bank pattern.
    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 32;
    static constexpr int kMfmaWaveGridM = 2;
    static constexpr int kMfmaWaveGridN = 4;
    static constexpr int kMfmaWaveTileM = 8;
    static constexpr int kMfmaWaveTileN = 2;
    static constexpr int kMfmaKStages = 1;          // BlockTileK = 32
    static constexpr int kMfmaBlocksPerCu = 2;
    static constexpr int kMfmaTileGroupM = 8;
    static constexpr int kMfmaLdsPadA = 0;
    static constexpr int kMfmaLdsPadB = 16;
};

template <>
struct KernelTuning<rocwmma::bfloat16_t> {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kScalarBlockTileK = 16;
    static constexpr int kScalarBlocksPerCu = 6;

    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 32;
    static constexpr int kMfmaWaveGridM = 2;
    static constexpr int kMfmaWaveGridN = 4;
    static constexpr int kMfmaWaveTileM = 8;
    static constexpr int kMfmaWaveTileN = 2;
    static constexpr int kMfmaKStages = 1;
    static constexpr int kMfmaBlocksPerCu = 2;
    static constexpr int kMfmaTileGroupM = 8;
    static constexpr int kMfmaLdsPadA = 0;
    static constexpr int kMfmaLdsPadB = 16;
};

template <>
struct KernelTuning<float> {
    static constexpr int kThreadsX = 16;
    static constexpr int kThreadsY = 16;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kScalarBlockTileK = 8;
    static constexpr int kScalarBlocksPerCu = 8;

    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 16;
    static constexpr int kMfmaWaveGridM = 4;
    static constexpr int kMfmaWaveGridN = 2;
    static constexpr int kMfmaWaveTileM = 4;
    static constexpr int kMfmaWaveTileN = 4;
    static constexpr int kMfmaKStages = 1;          // BlockTileK = 16
    static constexpr int kMfmaBlocksPerCu = 2;
    static constexpr int kMfmaTileGroupM = 8;
    static constexpr int kMfmaLdsPadA = 0;
    static constexpr int kMfmaLdsPadB = 4;
};

template <>
struct KernelTuning<double> {
    static constexpr int kThreadsX = 8;
    static constexpr int kThreadsY = 8;
    static constexpr int kThreadTileM = 4;
    static constexpr int kThreadTileN = 4;
    static constexpr int kScalarBlockTileK = 4;
    static constexpr int kScalarBlocksPerCu = 16;

    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 4;
    static constexpr int kMfmaWaveGridM = 2;
    static constexpr int kMfmaWaveGridN = 2;
    static constexpr int kMfmaWaveTileM = 2;
    static constexpr int kMfmaWaveTileN = 2;
    static constexpr int kMfmaKStages = 2;          // BlockTileK = 8
    static constexpr int kMfmaBlocksPerCu = 2;
    static constexpr int kMfmaTileGroupM = 8;
    static constexpr int kMfmaLdsPadA = 0;
    static constexpr int kMfmaLdsPadB = 0;
};

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

template <typename T>
constexpr int scalar_threads_per_block() {
    return KernelTuning<T>::kThreadsX * KernelTuning<T>::kThreadsY;
}

template <typename T>
constexpr int scalar_block_tile_m() {
    return KernelTuning<T>::kThreadsY * KernelTuning<T>::kThreadTileM;
}

template <typename T>
constexpr int scalar_block_tile_n() {
    return KernelTuning<T>::kThreadsX * KernelTuning<T>::kThreadTileN;
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
__host__ __device__ rocwmma::bfloat16_t cast_scalar<rocwmma::bfloat16_t>(double value) {
    return rocwmma::bfloat16_t(static_cast<float>(value));
}

template <typename T>
__host__ __device__ AccumT<T> to_accum(T value) {
    return static_cast<AccumT<T>>(value);
}

template <typename T>
int resolve_scalar_blocks_per_cu(const Args& args) {
    if(args.blocks_per_cu > 0) {
        return args.blocks_per_cu;
    }
    return KernelTuning<T>::kScalarBlocksPerCu;
}

template <typename T>
int resolve_mfma_blocks_per_cu(const Args& args) {
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

// Optimized MFMA GEMM kernel.
//   * Block tile is FragM * WaveGridM * WaveTileM  x  FragN * WaveGridN * WaveTileN
//     (e.g. 128x128 for fp16/bf16/fp32 with WaveTile=4x4, WaveGrid=2x2).
//   * BlockTileK = FragK * KStages so each outer K iteration drives multiple MFMAs
//     from one LDS fill (KStages > 1 amortises __syncthreads cost).
//   * Global -> LDS loads use 16-byte vector transfers (uint4) which map to
//     buffer_load_dwordx4 on CDNA3 and saturate the HBM path.
//   * LDS is double-buffered so the next K-tile is fetched while the current
//     MFMAs run; ds_reads inside mma_sync can overlap global_loads of the next
//     stage.
//   * Tile scheduling is swizzled in groups of TileGroupM rows so neighbouring
//     blocks share B columns in L2.
template <
    typename T,
    int FragM = KernelTuning<T>::kFragM,
    int FragN = KernelTuning<T>::kFragN,
    int FragK = KernelTuning<T>::kFragK,
    int WaveGridM = KernelTuning<T>::kMfmaWaveGridM,
    int WaveGridN = KernelTuning<T>::kMfmaWaveGridN,
    int WaveTileM = KernelTuning<T>::kMfmaWaveTileM,
    int WaveTileN = KernelTuning<T>::kMfmaWaveTileN,
    int KStages = KernelTuning<T>::kMfmaKStages,
    int TileGroupM = KernelTuning<T>::kMfmaTileGroupM,
    int LdsPadA = KernelTuning<T>::kMfmaLdsPadA,
    int LdsPadB = KernelTuning<T>::kMfmaLdsPadB>
__global__ __launch_bounds__(64 * WaveGridM * WaveGridN, KernelTuning<T>::kMfmaBlocksPerCu) void mfma_gemm_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    AccumT<T>* __restrict__ c,
    int m,
    int n,
    int k) {
    using Acc = AccumT<T>;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, FragM, FragN, FragK, T, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, FragM, FragN, FragK, T, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, FragM, FragN, FragK, Acc>;

    constexpr int kThreadsPerBlock = 64 * WaveGridM * WaveGridN;
    constexpr int kBlockTileM = FragM * WaveGridM * WaveTileM;
    constexpr int kBlockTileN = FragN * WaveGridN * WaveTileN;
    constexpr int kBlockTileK = FragK * KStages;
    static_assert(kThreadsPerBlock <= 1024, "MFMA kernel exceeds max threads per block");

    constexpr int kVecBytes = 16;
    constexpr int kVecElems = kVecBytes / static_cast<int>(sizeof(T));
    static_assert(kVecElems >= 1, "Vector element count must be >= 1");
    static_assert(kBlockTileK % kVecElems == 0, "BlockTileK must be divisible by vector elements");
    static_assert(kBlockTileN % kVecElems == 0, "BlockTileN must be divisible by vector elements");

    constexpr int kAVecsPerRow = kBlockTileK / kVecElems;
    constexpr int kBVecsPerRow = kBlockTileN / kVecElems;
    constexpr int kATotalVecs = kBlockTileM * kAVecsPerRow;
    constexpr int kBTotalVecs = kBlockTileK * kBVecsPerRow;
    constexpr int kATileStride = kBlockTileK + LdsPadA;
    constexpr int kBTileStride = kBlockTileN + LdsPadB;

    static_assert((kATileStride * static_cast<int>(sizeof(T))) % kVecBytes == 0, "A LDS stride must keep 16-byte alignment");
    static_assert((kBTileStride * static_cast<int>(sizeof(T))) % kVecBytes == 0, "B LDS stride must keep 16-byte alignment");

    __shared__ T a_tile[2][kBlockTileM * kATileStride];
    __shared__ T b_tile[2][kBlockTileK * kBTileStride];

    const int linear_tid = static_cast<int>(threadIdx.x);
    const int wave_id = linear_tid / 64;
    const int wave_row = wave_id / WaveGridN;
    const int wave_col = wave_id % WaveGridN;

    const int tiles_m = m / kBlockTileM;
    const int tiles_n = n / kBlockTileN;
    const int total_tiles = tiles_m * tiles_n;

    auto vec_load_a = [&](int buf, int k_base, int row_base) {
        #pragma unroll 1
        for(int vi = linear_tid; vi < kATotalVecs; vi += kThreadsPerBlock) {
            const int row_in_tile = vi / kAVecsPerRow;
            const int vec_in_row = vi % kAVecsPerRow;
            const int global_row = row_base + row_in_tile;
            const int global_k = k_base + vec_in_row * kVecElems;
            const uint4 v = *reinterpret_cast<const uint4*>(
                a + static_cast<std::size_t>(global_row) * static_cast<std::size_t>(k)
                  + static_cast<std::size_t>(global_k));
            *reinterpret_cast<uint4*>(
                &a_tile[buf][row_in_tile * kATileStride + vec_in_row * kVecElems]) = v;
        }
    };

    auto vec_load_b = [&](int buf, int k_base, int col_base) {
        #pragma unroll 1
        for(int vi = linear_tid; vi < kBTotalVecs; vi += kThreadsPerBlock) {
            const int row_in_tile = vi / kBVecsPerRow;
            const int vec_in_row = vi % kBVecsPerRow;
            const int global_k = k_base + row_in_tile;
            const int global_col = col_base + vec_in_row * kVecElems;
            const uint4 v = *reinterpret_cast<const uint4*>(
                b + static_cast<std::size_t>(global_k) * static_cast<std::size_t>(n)
                  + static_cast<std::size_t>(global_col));
            *reinterpret_cast<uint4*>(
                &b_tile[buf][row_in_tile * kBTileStride + vec_in_row * kVecElems]) = v;
        }
    };

    auto compute_stage = [&](int buf, FragC accum[WaveTileM][WaveTileN]) {
        #pragma unroll
        for(int ks = 0; ks < KStages; ++ks) {
            FragA a_frag[WaveTileM];
            FragB b_frag[WaveTileN];

            #pragma unroll
            for(int i = 0; i < WaveTileM; ++i) {
                const int a_row = (wave_row * WaveTileM + i) * FragM;
                rocwmma::load_matrix_sync(
                    a_frag[i],
                    &a_tile[buf][a_row * kATileStride + ks * FragK],
                    kATileStride);
            }
            #pragma unroll
            for(int j = 0; j < WaveTileN; ++j) {
                const int b_col = (wave_col * WaveTileN + j) * FragN;
                rocwmma::load_matrix_sync(
                    b_frag[j],
                    &b_tile[buf][ks * FragK * kBTileStride + b_col],
                    kBTileStride);
            }
            #pragma unroll
            for(int i = 0; i < WaveTileM; ++i) {
                #pragma unroll
                for(int j = 0; j < WaveTileN; ++j) {
                    rocwmma::mma_sync(accum[i][j], a_frag[i], b_frag[j], accum[i][j]);
                }
            }
        }
    };

    for(int tile_linear = static_cast<int>(blockIdx.x); tile_linear < total_tiles; tile_linear += static_cast<int>(gridDim.x)) {
        // Grouped-column swizzle: visit TileGroupM tile-rows at a time so neighbouring
        // blocks share the B column tile in L2.
        const int group_stride = TileGroupM * tiles_n;
        const int group_id = tile_linear / group_stride;
        const int first_tile_m = group_id * TileGroupM;
        const int group_rows = (first_tile_m + TileGroupM <= tiles_m)
            ? TileGroupM
            : (tiles_m - first_tile_m);
        const int tile_in_group = tile_linear % group_stride;
        const int tile_row = first_tile_m + (tile_in_group % group_rows);
        const int tile_col = tile_in_group / group_rows;
        const int row_base = tile_row * kBlockTileM;
        const int col_base = tile_col * kBlockTileN;

        FragC accum[WaveTileM][WaveTileN];
        #pragma unroll
        for(int i = 0; i < WaveTileM; ++i) {
            #pragma unroll
            for(int j = 0; j < WaveTileN; ++j) {
                rocwmma::fill_fragment(accum[i][j], static_cast<Acc>(0));
            }
        }

        int buf = 0;
        // Prologue: load the first K-tile into buf 0.
        vec_load_a(buf, 0, row_base);
        vec_load_b(buf, 0, col_base);
        __syncthreads();

        // Main pipelined loop: while computing on buf, prefetch next tile into the
        // other buffer. Requires a __syncthreads after both compute and fill to
        // keep the buffers consistent across waves.
        for(int k_base = 0; k_base + kBlockTileK < k; k_base += kBlockTileK) {
            const int next_buf = buf ^ 1;
            const int next_k_base = k_base + kBlockTileK;

            vec_load_a(next_buf, next_k_base, row_base);
            vec_load_b(next_buf, next_k_base, col_base);
            compute_stage(buf, accum);
            __syncthreads();
            buf = next_buf;
        }

        // Epilogue: compute on the final loaded buffer.
        compute_stage(buf, accum);
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < WaveTileM; ++i) {
            const int global_row = row_base + (wave_row * WaveTileM + i) * FragM;
            #pragma unroll
            for(int j = 0; j < WaveTileN; ++j) {
                const int global_col = col_base + (wave_col * WaveTileN + j) * FragN;
                rocwmma::store_matrix_sync(
                    &c[static_cast<std::size_t>(global_row) * static_cast<std::size_t>(n)
                       + static_cast<std::size_t>(global_col)],
                    accum[i][j],
                    n,
                    rocwmma::mem_row_major
                );
            }
        }
    }
}

template <typename T>
bool can_use_mfma(const Args& args) {
    return (args.m % mfma_block_tile_m<T>() == 0)
        && (args.n % mfma_block_tile_n<T>() == 0)
        && (args.k % mfma_block_tile_k<T>() == 0);
}

template <typename T>
double run_kernel(const Args& args, bool* used_mfma, int* active_blocks_per_cu) {
    check_hip(hipSetDevice(args.device_id), "hipSetDevice");

    using Acc = AccumT<T>;

    const std::size_t a_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.k);
    const std::size_t b_count = static_cast<std::size_t>(args.k) * static_cast<std::size_t>(args.n);
    const std::size_t c_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.n);

    T* a = nullptr;
    T* b = nullptr;
    Acc* c = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, args.device_id), "hipGetDeviceProperties");

    check_hip(hipMalloc(&reinterpret_cast<void*&>(a), a_count * sizeof(T)), "hipMalloc(a)");
    check_hip(hipMalloc(&reinterpret_cast<void*&>(b), b_count * sizeof(T)), "hipMalloc(b)");
    check_hip(hipMalloc(&reinterpret_cast<void*&>(c), c_count * sizeof(Acc)), "hipMalloc(c)");

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
    hipLaunchKernelGGL(fill_kernel<Acc>, dim3(init_blocks), dim3(kInitThreads), 0, stream, c, c_count, static_cast<Acc>(0));
    check_hip(hipGetLastError(), "fill_kernel(c)");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(init)");

    const bool use_mfma = can_use_mfma<T>(args);
    // We only benchmark MFMA kernel
    assert(use_mfma && "MFMA kernel is not properly called");

    if(used_mfma != nullptr) {
        *used_mfma = use_mfma;
    }
    if(active_blocks_per_cu != nullptr) {
        *active_blocks_per_cu = 0;
    }

    int max_active_blocks_per_cu = 0;
    check_hip(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_per_cu,
            mfma_gemm_kernel<T>,
            mfma_threads_per_block<T>(),
            0),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor");
    if(active_blocks_per_cu != nullptr) {
        *active_blocks_per_cu = max_active_blocks_per_cu;
    }

    auto launch_once = [&]() {
        const int blocks = std::max(
            1,
            std::min(
                std::max(1, (args.m / mfma_block_tile_m<T>()) * (args.n / mfma_block_tile_n<T>())),
                props.multiProcessorCount * std::max(1, resolve_mfma_blocks_per_cu<T>(args))
            )
        );
        hipLaunchKernelGGL(mfma_gemm_kernel<T>, dim3(blocks), dim3(mfma_threads_per_block<T>()), 0, stream, a, b, c, args.m, args.n, args.k);
        check_hip(hipGetLastError(), "mfma_gemm_kernel");
        return;
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
    bool used_mfma = false;
    int active_blocks_per_cu = 0;
    const double elapsed_s = run_kernel<T>(args, &used_mfma, &active_blocks_per_cu);
    std::cout << "{\"status\":\"ok\",\"raw_metrics\":{\"elapsed_s\":" << elapsed_s
              << "},\"metadata\":{\"implementation\":\"" << (used_mfma ? "rocwmma_mfma" : "hip_scalar")
              << "\",\"threads_per_block\":" << (used_mfma ? mfma_threads_per_block<T>() : scalar_threads_per_block<T>())
              << ",\"tile_m\":" << (used_mfma ? mfma_block_tile_m<T>() : scalar_block_tile_m<T>())
              << ",\"tile_n\":" << (used_mfma ? mfma_block_tile_n<T>() : scalar_block_tile_n<T>())
              << ",\"tile_k\":" << (used_mfma ? mfma_block_tile_k<T>() : KernelTuning<T>::kScalarBlockTileK)
              << ",\"blocks_per_cu\":"
              << (used_mfma ? resolve_mfma_blocks_per_cu<T>(args) : resolve_scalar_blocks_per_cu<T>(args))
              << ",\"active_blocks_per_cu\":" << active_blocks_per_cu;
    if(used_mfma) {
        std::cout << ",\"wave_grid_m\":" << KernelTuning<T>::kMfmaWaveGridM
                  << ",\"wave_grid_n\":" << KernelTuning<T>::kMfmaWaveGridN
                  << ",\"wave_tile_m\":" << KernelTuning<T>::kMfmaWaveTileM
                  << ",\"wave_tile_n\":" << KernelTuning<T>::kMfmaWaveTileN
                  << ",\"frag_m\":" << KernelTuning<T>::kFragM
                  << ",\"frag_n\":" << KernelTuning<T>::kFragN
                  << ",\"frag_k\":" << KernelTuning<T>::kFragK
                  << ",\"k_stages\":" << KernelTuning<T>::kMfmaKStages
                  << ",\"tile_group_m\":" << KernelTuning<T>::kMfmaTileGroupM;
    } else {
        std::cout << ",\"thread_tile_m\":" << KernelTuning<T>::kThreadTileM
                  << ",\"thread_tile_n\":" << KernelTuning<T>::kThreadTileN;
    }
    std::cout << "}}" << std::endl;
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
            return emit_success<rocwmma::float16_t>(args);
        }
        if(args.dtype == "bfloat16") {
            return emit_success<rocwmma::bfloat16_t>(args);
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
