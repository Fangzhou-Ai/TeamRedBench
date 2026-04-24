#pragma once

#include "mfu_gemm_common.hip.hpp"

namespace teamredbench::native::mfu_gemm {

template <>
struct KernelTuning<rocwmma::float16_t> {
    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 32;
    static constexpr int kMfmaWaveGridM = 2;
    static constexpr int kMfmaWaveGridN = 4;
    static constexpr int kMfmaWaveTileM = 4;
    static constexpr int kMfmaWaveTileN = 4;
    static constexpr int kMfmaKStages = 1;
    static constexpr int kMfmaBlocksPerCu = 4;
    static constexpr int kMfmaTileGroupM = 1;
    static constexpr int kMfmaLdsPadA = 0;
    static constexpr int kMfmaLdsPadB = 0;
};

template <>
struct KernelTuning<rocwmma::bfloat16_t> {
    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 32;
    static constexpr int kMfmaWaveGridM = 2;
    static constexpr int kMfmaWaveGridN = 4;
    static constexpr int kMfmaWaveTileM = 4;
    static constexpr int kMfmaWaveTileN = 4;
    static constexpr int kMfmaKStages = 1;
    static constexpr int kMfmaBlocksPerCu = 4;
    static constexpr int kMfmaTileGroupM = 2;
    static constexpr int kMfmaLdsPadA = 0;
    static constexpr int kMfmaLdsPadB = 0;
};

// K stride of a packed B tile column. Packing B with K contiguous per N column
// lets the raw MFMA path load each B operand vector with one 16-byte LDS read.
template <typename T>
constexpr int packed_b_tile_stride() {
    return mfma_block_tile_k<T>() + KernelTuning<T>::kMfmaLdsPadB;
}

template <typename T>
constexpr std::size_t packed_b_tile_elements() {
    return static_cast<std::size_t>(mfma_block_tile_n<T>()) * static_cast<std::size_t>(packed_b_tile_stride<T>());
}

template <typename T>
__device__ __forceinline__ void global_load_vec16_to_lds(const T* src, T* dst) {
    auto* global_ptr = (__attribute__((address_space(1))) void*)(const_cast<T*>(src));
    auto* lds_ptr = (__attribute__((address_space(3))) void*)(dst);
    __builtin_amdgcn_global_load_lds(global_ptr, lds_ptr, 16u, 0, 0u);
}

using MfmaInputVec4 = uint32_t __attribute__((ext_vector_type(4)));
using MfmaAccumVec4 = float __attribute__((ext_vector_type(4)));

template <typename T>
__device__ __forceinline__ MfmaAccumVec4 mfma_16x16x32(MfmaInputVec4 a, MfmaInputVec4 b, MfmaAccumVec4 c);

template <>
__device__ __forceinline__ MfmaAccumVec4 mfma_16x16x32<rocwmma::float16_t>(
    MfmaInputVec4 a,
    MfmaInputVec4 b,
    MfmaAccumVec4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, 0);
}

template <>
__device__ __forceinline__ MfmaAccumVec4 mfma_16x16x32<rocwmma::bfloat16_t>(
    MfmaInputVec4 a,
    MfmaInputVec4 b,
    MfmaAccumVec4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
}

// Pack B once into tile-local K-contiguous columns so the timed kernel avoids
// strided B operand gathers in the hot MFMA loop.
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
    int LdsPadB = KernelTuning<T>::kMfmaLdsPadB>
__global__ void pack_b_tiles_kernel(const T* __restrict__ b, T* __restrict__ b_packed, int n, int k) {
    constexpr int kBlockTileN = FragN * WaveGridN * WaveTileN;
    constexpr int kBlockTileK = FragK * KStages;
    constexpr int kVecBytes = 16;
    constexpr int kVecElems = kVecBytes / static_cast<int>(sizeof(T));
    constexpr int kPackedStride = kBlockTileK + LdsPadB;
    constexpr int kVecsPerCol = kBlockTileK / kVecElems;
    constexpr int kTotalVecs = kBlockTileN * kVecsPerCol;

    static_assert((kPackedStride * static_cast<int>(sizeof(T))) % kVecBytes == 0,
                  "Packed B stride must keep 16-byte alignment");
    static_assert(kBlockTileK % kVecElems == 0, "BlockTileK must be divisible by vector elements");

    const int tiles_n = n / kBlockTileN;
    const int tiles_k = k / kBlockTileK;
    const int total_tiles = tiles_n * tiles_k;
    const int tile_id = static_cast<int>(blockIdx.x);
    if(tile_id >= total_tiles) {
        return;
    }

    const int tile_k = tile_id / tiles_n;
    const int tile_n = tile_id % tiles_n;
    const int k_base = tile_k * kBlockTileK;
    const int n_base = tile_n * kBlockTileN;
    const std::size_t packed_tile_offset = static_cast<std::size_t>(tile_id) * packed_b_tile_elements<T>();

    for(int vi = static_cast<int>(threadIdx.x); vi < kTotalVecs; vi += static_cast<int>(blockDim.x)) {
        const int col_in_tile = vi / kVecsPerCol;
        const int vec_in_col = vi % kVecsPerCol;
        #pragma unroll
        for(int e = 0; e < kVecElems; ++e) {
            b_packed[packed_tile_offset
                     + static_cast<std::size_t>(col_in_tile) * static_cast<std::size_t>(kPackedStride)
                     + static_cast<std::size_t>(vec_in_col * kVecElems + e)] =
                b[static_cast<std::size_t>(k_base + vec_in_col * kVecElems + e) * static_cast<std::size_t>(n)
                  + static_cast<std::size_t>(n_base + col_in_tile)];
        }
    }
}

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
__global__ __launch_bounds__(64 * WaveGridM * WaveGridN, KernelTuning<T>::kMfmaBlocksPerCu) void mfma_gemm_lowp_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b_packed,
    T* __restrict__ c,
    int m,
    int n,
    int k) {
    static_assert(std::is_same_v<T, rocwmma::float16_t> || std::is_same_v<T, rocwmma::bfloat16_t>,
                  "Low-precision kernel only supports fp16 and bf16");
    static_assert(FragM == 16 && FragN == 16 && FragK == 32,
                  "Raw low-precision MFMA path expects 16x16x32 fragments");

    constexpr int kThreadsPerBlock = 64 * WaveGridM * WaveGridN;
    constexpr int kBlockTileM = FragM * WaveGridM * WaveTileM;
    constexpr int kBlockTileN = FragN * WaveGridN * WaveTileN;
    constexpr int kBlockTileK = FragK * KStages;

    constexpr int kVecBytes = 16;
    constexpr int kVecElems = kVecBytes / static_cast<int>(sizeof(T));
    constexpr int kATileStride = kBlockTileK + LdsPadA;
    constexpr int kBTileStride = kBlockTileK + LdsPadB;
    constexpr int kWavesPerBlock = kThreadsPerBlock / 64;

    static_assert(kBlockTileK % kVecElems == 0, "BlockTileK must be divisible by vector elements");
    static_assert(kBlockTileN % kVecElems == 0, "BlockTileN must be divisible by vector elements");
    static_assert((kATileStride * static_cast<int>(sizeof(T))) % kVecBytes == 0,
                  "A LDS stride must keep 16-byte alignment");
    static_assert((kBTileStride * static_cast<int>(sizeof(T))) % kVecBytes == 0,
                  "B LDS stride must keep 16-byte alignment");

    constexpr int kPipelineStages = 2;
    __shared__ T a_tile[kPipelineStages][kBlockTileM * kATileStride];
    __shared__ T b_tile[kPipelineStages][kBlockTileN * kBTileStride];

    const int linear_tid = static_cast<int>(threadIdx.x);
    const int wave_id = linear_tid / 64;
    const int wave_row = wave_id / WaveGridN;
    const int wave_col = wave_id % WaveGridN;

    const int tiles_m = m / kBlockTileM;
    const int tiles_n = n / kBlockTileN;
    const int total_tiles = tiles_m * tiles_n;

    // A is strided in global memory, so each thread issues its own direct
    // global-to-LDS transfer. Packed B is contiguous and can use wave-coalesced
    // direct-to-LDS transfers below.
    constexpr int kAVecsPerRow = kBlockTileK / kVecElems;
    constexpr int kATotalVecs = kBlockTileM * kAVecsPerRow;

    constexpr int kBVecsPerCol = kBlockTileK / kVecElems;
    constexpr int kBTotalVecs = kBlockTileN * kBVecsPerCol;

    auto vec_load_a = [&](int buf, int k_base, int row_base) {
        #pragma unroll 1
        for(int vi = linear_tid; vi < kATotalVecs; vi += kThreadsPerBlock) {
            const int row_in_tile = vi / kAVecsPerRow;
            const int vec_in_row = vi % kAVecsPerRow;
            const int global_row = row_base + row_in_tile;
            const int global_k = k_base + vec_in_row * kVecElems;
            const T* global_src = a + static_cast<std::size_t>(global_row) * static_cast<std::size_t>(k)
                + static_cast<std::size_t>(global_k);
            T* lds_dst = &a_tile[buf][row_in_tile * kATileStride + vec_in_row * kVecElems];
            global_load_vec16_to_lds(global_src, lds_dst);
        }
    };

    auto vec_load_b = [&](int buf, int k_base, int col_base) {
        const int tile_col = col_base / kBlockTileN;
        const int tile_k = k_base / kBlockTileK;
        const std::size_t packed_tile_offset =
            (static_cast<std::size_t>(tile_k) * static_cast<std::size_t>(tiles_n) + static_cast<std::size_t>(tile_col))
            * packed_b_tile_elements<T>();

        #pragma unroll 1
        for(int vi = linear_tid; vi < kBTotalVecs; vi += kThreadsPerBlock) {
            const int col_in_tile = vi / kBVecsPerCol;
            const int vec_in_col = vi % kBVecsPerCol;
            const T* global_src = b_packed + packed_tile_offset
                + static_cast<std::size_t>(col_in_tile) * static_cast<std::size_t>(kBTileStride)
                + static_cast<std::size_t>(vec_in_col * kVecElems);
            T* lds_dst = &b_tile[buf][col_in_tile * kBTileStride + vec_in_col * kVecElems];
            global_load_vec16_to_lds(global_src, lds_dst);
        }
    };

    auto compute_stage = [&](int buf, MfmaAccumVec4 accum[WaveTileM][WaveTileN]) {
        const int lane = linear_tid & 63;
        const int a_lane_m = lane & 15;
        const int a_lane_k = lane >> 4;
        const int b_lane_n = lane & 15;
        const int b_lane_k = lane >> 4;

        #pragma unroll
        for(int ks = 0; ks < KStages; ++ks) {
            MfmaInputVec4 a_frag[WaveTileM];
            MfmaInputVec4 b_frag[WaveTileN];

            #pragma unroll
            for(int i = 0; i < WaveTileM; ++i) {
                const int a_row = (wave_row * WaveTileM + i) * FragM;
                a_frag[i] = *reinterpret_cast<const MfmaInputVec4*>(
                    &a_tile[buf][(a_row + a_lane_m) * kATileStride + ks * FragK + a_lane_k * 8]);
            }
            #pragma unroll
            for(int j = 0; j < WaveTileN; ++j) {
                const int b_col = (wave_col * WaveTileN + j) * FragN;
                b_frag[j] = *reinterpret_cast<const MfmaInputVec4*>(
                    &b_tile[buf][(b_col + b_lane_n) * kBTileStride + ks * FragK + b_lane_k * 8]);
            }
            #pragma unroll
            for(int i = 0; i < WaveTileM; ++i) {
                #pragma unroll
                for(int j = 0; j < WaveTileN; ++j) {
                    accum[i][j] = mfma_16x16x32<T>(a_frag[i], b_frag[j], accum[i][j]);
                }
            }
        }
    };

    for(int tile_linear = static_cast<int>(blockIdx.x); tile_linear < total_tiles; tile_linear += static_cast<int>(gridDim.x)) {
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

        MfmaAccumVec4 accum[WaveTileM][WaveTileN];
        #pragma unroll
        for(int i = 0; i < WaveTileM; ++i) {
            #pragma unroll
            for(int j = 0; j < WaveTileN; ++j) {
                accum[i][j] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        // Pipeline global-to-LDS staging ahead of the MFMA loop.
        const int num_k_stages = k / kBlockTileK;

        int load_stage = 0;
        while(load_stage < kPipelineStages - 1 && load_stage < num_k_stages) {
            vec_load_a(load_stage, load_stage * kBlockTileK, row_base);
            vec_load_b(load_stage, load_stage * kBlockTileK, col_base);
            ++load_stage;
        }
        __syncthreads();

        int compute_stage_idx = 0;
        for(; compute_stage_idx + (kPipelineStages - 1) < num_k_stages; ++compute_stage_idx) {
            const int load_buf = load_stage % kPipelineStages;
            const int compute_buf = compute_stage_idx % kPipelineStages;
            vec_load_a(load_buf, load_stage * kBlockTileK, row_base);
            vec_load_b(load_buf, load_stage * kBlockTileK, col_base);
            compute_stage(compute_buf, accum);
            __syncthreads();
            ++load_stage;
        }

        for(; compute_stage_idx < num_k_stages; ++compute_stage_idx) {
            const int compute_buf = compute_stage_idx % kPipelineStages;
            compute_stage(compute_buf, accum);
            __syncthreads();
        }

        const int c_lane_m = ((linear_tid >> 4) & 3) * 4;
        const int c_lane_n = linear_tid & 15;

        #pragma unroll
        for(int i = 0; i < WaveTileM; ++i) {
            const int global_row = row_base + (wave_row * WaveTileM + i) * FragM;
            #pragma unroll
            for(int j = 0; j < WaveTileN; ++j) {
                const int global_col = col_base + (wave_col * WaveTileN + j) * FragN;
                #pragma unroll
                for(int element = 0; element < 4; ++element) {
                    c[static_cast<std::size_t>(global_row + c_lane_m + element) * static_cast<std::size_t>(n)
                      + static_cast<std::size_t>(global_col + c_lane_n)] =
                        cast_output<T>(accum[i][j][element]);
                }
            }
        }
    }
}

template <typename T>
inline bool can_use_low_precision_mfma(const Args& args) {
    return (args.m % mfma_block_tile_m<T>() == 0)
        && (args.n % mfma_block_tile_n<T>() == 0)
        && (args.k % mfma_block_tile_k<T>() == 0);
}

template <typename T>
inline double run_low_precision_kernel(const Args& args, int* active_blocks_per_cu) {
    check_hip(hipSetDevice(args.device_id), "hipSetDevice");

    const std::size_t a_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.k);
    const std::size_t b_count = static_cast<std::size_t>(args.k) * static_cast<std::size_t>(args.n);
    const std::size_t c_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.n);
    const std::size_t packed_b_count =
        static_cast<std::size_t>(args.k / mfma_block_tile_k<T>())
        * static_cast<std::size_t>(args.n / mfma_block_tile_n<T>())
        * packed_b_tile_elements<T>();

    T* a = nullptr;
    T* b = nullptr;
    T* b_packed = nullptr;
    T* c = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, args.device_id), "hipGetDeviceProperties");

    check_hip(hipMalloc(&reinterpret_cast<void*&>(a), a_count * sizeof(T)), "hipMalloc(a)");
    check_hip(hipMalloc(&reinterpret_cast<void*&>(b), b_count * sizeof(T)), "hipMalloc(b)");
    check_hip(hipMalloc(&reinterpret_cast<void*&>(b_packed), packed_b_count * sizeof(T)), "hipMalloc(b_packed)");
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
    hipLaunchKernelGGL(fill_kernel<T>, dim3(init_blocks), dim3(kInitThreads), 0, stream, c, c_count, static_cast<T>(0));
    check_hip(hipGetLastError(), "fill_kernel(c)");

    if(!can_use_low_precision_mfma<T>(args)) {
        fail("low-precision MFMA kernel requires m, n, and k to be divisible by the configured tile shape");
    }

    const int pack_tiles = (args.k / mfma_block_tile_k<T>()) * (args.n / mfma_block_tile_n<T>());
    constexpr int kPackThreads = 256;
    hipLaunchKernelGGL(
        pack_b_tiles_kernel<T>,
        dim3(static_cast<unsigned int>(std::max(1, pack_tiles))),
        dim3(kPackThreads),
        0,
        stream,
        b,
        b_packed,
        args.n,
        args.k);
    check_hip(hipGetLastError(), "pack_b_tiles_kernel");
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(init)");

    int max_active_blocks_per_cu = 0;
    check_hip(
        hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_per_cu,
            mfma_gemm_lowp_kernel<T>,
            mfma_threads_per_block<T>(),
            0),
        "hipOccupancyMaxActiveBlocksPerMultiprocessor(lowp)");
    if(active_blocks_per_cu != nullptr) {
        *active_blocks_per_cu = max_active_blocks_per_cu;
    }

    auto launch_once = [&]() {
        const int tile_blocks = std::max(1, (args.m / mfma_block_tile_m<T>()) * (args.n / mfma_block_tile_n<T>()));
        const int blocks = std::max(
            1,
            std::min(
                tile_blocks,
                props.multiProcessorCount * std::max(1, resolve_mfma_blocks_per_cu<T>(args))
            )
        );
        hipLaunchKernelGGL(
            mfma_gemm_lowp_kernel<T>,
            dim3(blocks),
            dim3(mfma_threads_per_block<T>()),
            0,
            stream,
            a,
            b_packed,
            c,
            args.m,
            args.n,
            args.k);
        check_hip(hipGetLastError(), "mfma_gemm_lowp_kernel");
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
    if(b_packed != nullptr) {
        ignore_hip(hipFree(b_packed));
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
inline int emit_low_precision_success(const Args& args) {
    int active_blocks_per_cu = 0;
    const double elapsed_s = run_low_precision_kernel<T>(args, &active_blocks_per_cu);
    std::cout << "{\"status\":\"ok\",\"raw_metrics\":{\"elapsed_s\":" << elapsed_s
              << "},\"metadata\":{\"implementation\":\"raw_mfma_col_packed_b\""
              << ",\"threads_per_block\":" << mfma_threads_per_block<T>()
              << ",\"tile_m\":" << mfma_block_tile_m<T>()
              << ",\"tile_n\":" << mfma_block_tile_n<T>()
              << ",\"tile_k\":" << mfma_block_tile_k<T>()
              << ",\"blocks_per_cu\":" << resolve_mfma_blocks_per_cu<T>(args)
              << ",\"active_blocks_per_cu\":" << active_blocks_per_cu
              << ",\"wave_grid_m\":" << KernelTuning<T>::kMfmaWaveGridM
              << ",\"wave_grid_n\":" << KernelTuning<T>::kMfmaWaveGridN
              << ",\"wave_tile_m\":" << KernelTuning<T>::kMfmaWaveTileM
              << ",\"wave_tile_n\":" << KernelTuning<T>::kMfmaWaveTileN
              << ",\"frag_m\":" << KernelTuning<T>::kFragM
              << ",\"frag_n\":" << KernelTuning<T>::kFragN
              << ",\"frag_k\":" << KernelTuning<T>::kFragK
              << ",\"k_stages\":" << KernelTuning<T>::kMfmaKStages
              << ",\"tile_group_m\":" << KernelTuning<T>::kMfmaTileGroupM
              << "}}" << std::endl;
    return 0;
}

}  // namespace teamredbench::native::mfu_gemm
