#pragma once

#include "mfu_gemm_common.hip.hpp"

namespace teamredbench::native::mfu_gemm {

template <>
struct KernelTuning<rocwmma::float16_t> {
    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 32;
    static constexpr int kMfmaWaveGridM = 4;
    static constexpr int kMfmaWaveGridN = 2;
    static constexpr int kMfmaWaveTileM = 4;
    static constexpr int kMfmaWaveTileN = 4;
    static constexpr int kMfmaKStages = 1;
    static constexpr int kMfmaBlocksPerCu = 3;
    static constexpr int kMfmaTileGroupM = 8;
    static constexpr int kMfmaLdsPadA = 16;
    static constexpr int kMfmaLdsPadB = 0;
};

template <>
struct KernelTuning<rocwmma::bfloat16_t> {
    static constexpr int kFragM = 16;
    static constexpr int kFragN = 16;
    static constexpr int kFragK = 32;
    static constexpr int kMfmaWaveGridM = 4;
    static constexpr int kMfmaWaveGridN = 2;
    static constexpr int kMfmaWaveTileM = 4;
    static constexpr int kMfmaWaveTileN = 4;
    static constexpr int kMfmaKStages = 1;
    static constexpr int kMfmaBlocksPerCu = 3;
    static constexpr int kMfmaTileGroupM = 8;
    static constexpr int kMfmaLdsPadA = 16;
    static constexpr int kMfmaLdsPadB = 0;
};

// Row-stride (along N) of a packed B tile. Tiles are stored row-major so that
// rocWMMA's row_major matrix_b fragments read them cleanly; the original
// col_major layout hit a correctness bug in rocWMMA's 16x16x32 fp16/bf16 path.
template <typename T>
constexpr int packed_b_tile_stride() {
    return mfma_block_tile_n<T>() + KernelTuning<T>::kMfmaLdsPadB;
}

template <typename T>
constexpr std::size_t packed_b_tile_elements() {
    return static_cast<std::size_t>(mfma_block_tile_k<T>()) * static_cast<std::size_t>(packed_b_tile_stride<T>());
}

template <typename T>
__device__ __forceinline__ void global_load_vec16_to_lds(const T* src, T* dst) {
    auto* global_ptr = (__attribute__((address_space(1))) void*)(const_cast<T*>(src));
    auto* lds_ptr = (__attribute__((address_space(3))) void*)(dst);
    __builtin_amdgcn_global_load_lds(global_ptr, lds_ptr, 16u, 0, 0u);
}

// Pack B once into tile-local row-major blocks so the timed kernel can use
// contiguous global_load_lds transfers and a row_major matrix_b fragment load
// without paying the gather/stride cost every iteration.
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
    constexpr int kPackedStride = kBlockTileN + LdsPadB;
    constexpr int kVecsPerRow = kBlockTileN / kVecElems;
    constexpr int kTotalVecs = kBlockTileK * kVecsPerRow;

    static_assert((kPackedStride * static_cast<int>(sizeof(T))) % kVecBytes == 0,
                  "Packed B stride must keep 16-byte alignment");
    static_assert(kBlockTileN % kVecElems == 0, "BlockTileN must be divisible by vector elements");

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
        const int row_in_tile = vi / kVecsPerRow;
        const int vec_in_row = vi % kVecsPerRow;
        const uint4 v = *reinterpret_cast<const uint4*>(
            b + static_cast<std::size_t>(k_base + row_in_tile) * static_cast<std::size_t>(n)
              + static_cast<std::size_t>(n_base + vec_in_row * kVecElems));
        *reinterpret_cast<uint4*>(
            &b_packed[packed_tile_offset
                      + static_cast<std::size_t>(row_in_tile) * static_cast<std::size_t>(kPackedStride)
                      + static_cast<std::size_t>(vec_in_row * kVecElems)]) = v;
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
    using Acc = AccumT<T>;
    using FragA = rocwmma::fragment<rocwmma::matrix_a, FragM, FragN, FragK, T, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, FragM, FragN, FragK, T, rocwmma::row_major>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, FragM, FragN, FragK, Acc>;
    using FragOut = rocwmma::fragment<rocwmma::accumulator, FragM, FragN, FragK, T, rocwmma::row_major>;

    constexpr int kThreadsPerBlock = 64 * WaveGridM * WaveGridN;
    constexpr int kBlockTileM = FragM * WaveGridM * WaveTileM;
    constexpr int kBlockTileN = FragN * WaveGridN * WaveTileN;
    constexpr int kBlockTileK = FragK * KStages;

    constexpr int kVecBytes = 16;
    constexpr int kVecElems = kVecBytes / static_cast<int>(sizeof(T));
    constexpr int kATileStride = kBlockTileK + LdsPadA;
    constexpr int kBTileStride = kBlockTileN + LdsPadB;
    constexpr int kWavesPerBlock = kThreadsPerBlock / 64;
    constexpr int kBytesPerPass = 64 * kVecBytes;

    static_assert(kBlockTileK % kVecElems == 0, "BlockTileK must be divisible by vector elements");
    static_assert(kBlockTileN % kVecElems == 0, "BlockTileN must be divisible by vector elements");
    static_assert((kATileStride * static_cast<int>(sizeof(T))) % kVecBytes == 0,
                  "A LDS stride must keep 16-byte alignment");
    static_assert((kBTileStride * static_cast<int>(sizeof(T))) % kVecBytes == 0,
                  "B LDS stride must keep 16-byte alignment");

    __shared__ T a_tile[2][kBlockTileM * kATileStride];
    __shared__ T b_tile[2][kBlockTileK * kBTileStride];

    const int linear_tid = static_cast<int>(threadIdx.x);
    const int wave_id = linear_tid / 64;
    const int wave_row = wave_id / WaveGridN;
    const int wave_col = wave_id % WaveGridN;

    const int tiles_m = m / kBlockTileM;
    const int tiles_n = n / kBlockTileN;
    const int total_tiles = tiles_m * tiles_n;

    // A keeps padded LDS (LdsPadA > 0) to avoid bank conflicts during the MFMA
    // load_matrix_sync reads, so it cannot use the wave-uniform-LDS intrinsic
    // path used for B. Plain uint4 loads are fast enough given the padded layout.
    constexpr int kAVecsPerRow = kBlockTileK / kVecElems;
    constexpr int kATotalVecs = kBlockTileM * kAVecsPerRow;

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

    // B uses wave-collective global_load_lds. Each wave issues 1024-byte loads
    // that map 1:1 onto contiguous rows of the packed tile (requires LdsPadB=0
    // and kBlockTileN chosen so (kBlockTileN * sizeof(T)) divides 64 * kVecBytes).
    static_assert(LdsPadB == 0, "intrinsic B path requires LdsPadB=0 so rows are contiguous in LDS");
    constexpr int kBRowBytes = kBlockTileN * static_cast<int>(sizeof(T));
    static_assert(kBytesPerPass % kBRowBytes == 0, "kBlockTileN*sizeof(T) must divide 64*16");
    constexpr int kBRowsPerPass = kBytesPerPass / kBRowBytes;
    constexpr int kBTotalPasses = kBlockTileK / kBRowsPerPass;
    static_assert(kBlockTileK % kBRowsPerPass == 0, "kBlockTileK must be a multiple of kBRowsPerPass");
    constexpr int kBPassesPerWave = kBTotalPasses / kWavesPerBlock;
    constexpr int kBTailPasses = kBTotalPasses - kBPassesPerWave * kWavesPerBlock;

    auto vec_load_b = [&](int buf, int k_base, int col_base) {
        const int tile_col = col_base / kBlockTileN;
        const int tile_k = k_base / kBlockTileK;
        const std::size_t packed_tile_offset =
            (static_cast<std::size_t>(tile_k) * static_cast<std::size_t>(tiles_n) + static_cast<std::size_t>(tile_col))
            * packed_b_tile_elements<T>();

        const int lane = linear_tid & 63;
        const int wave_id_local = linear_tid / 64;

        #pragma unroll
        for(int p = 0; p < kBPassesPerWave; ++p) {
            const int row_in_tile = (wave_id_local * kBPassesPerWave + p) * kBRowsPerPass;
            const T* global_src = b_packed + packed_tile_offset
                + static_cast<std::size_t>(row_in_tile) * static_cast<std::size_t>(kBlockTileN)
                + static_cast<std::size_t>(lane) * static_cast<std::size_t>(kVecElems);
            T* lds_dst = &b_tile[buf][row_in_tile * kBTileStride + lane * kVecElems];
            global_load_vec16_to_lds(global_src, lds_dst);
        }
        if constexpr(kBTailPasses > 0) {
            if(wave_id_local < kBTailPasses) {
                const int row_in_tile = (kBPassesPerWave * kWavesPerBlock + wave_id_local) * kBRowsPerPass;
                const T* global_src = b_packed + packed_tile_offset
                    + static_cast<std::size_t>(row_in_tile) * static_cast<std::size_t>(kBlockTileN)
                    + static_cast<std::size_t>(lane) * static_cast<std::size_t>(kVecElems);
                T* lds_dst = &b_tile[buf][row_in_tile * kBTileStride + lane * kVecElems];
                global_load_vec16_to_lds(global_src, lds_dst);
            }
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
        // Let the backend interleave LDS reads and MFMA issue more aggressively
        // in the hot GEMM loop.
        __builtin_amdgcn_iglp_opt(0);

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
        vec_load_a(buf, 0, row_base);
        vec_load_b(buf, 0, col_base);
        __syncthreads();

        for(int k_base = 0; k_base + kBlockTileK < k; k_base += kBlockTileK) {
            const int next_buf = buf ^ 1;
            const int next_k_base = k_base + kBlockTileK;

            vec_load_a(next_buf, next_k_base, row_base);
            vec_load_b(next_buf, next_k_base, col_base);
            compute_stage(buf, accum);
            __syncthreads();
            buf = next_buf;
        }

        compute_stage(buf, accum);
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < WaveTileM; ++i) {
            const int global_row = row_base + (wave_row * WaveTileM + i) * FragM;
            #pragma unroll
            for(int j = 0; j < WaveTileN; ++j) {
                const int global_col = col_base + (wave_col * WaveTileN + j) * FragN;
                T* out_ptr = &c[static_cast<std::size_t>(global_row) * static_cast<std::size_t>(n)
                                + static_cast<std::size_t>(global_col)];
                FragOut out_frag;
                #pragma unroll
                for(int element = 0; element < out_frag.num_elements; ++element) {
                    out_frag.x[element] = cast_output<T>(accum[i][j].x[element]);
                }
                rocwmma::store_matrix_sync(out_ptr, out_frag, n);
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
        hipLaunchKernelGGL(mfma_gemm_lowp_kernel<T>, dim3(blocks), dim3(mfma_threads_per_block<T>()), 0, stream, a, b_packed, c, args.m, args.n, args.k);
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
              << "},\"metadata\":{\"implementation\":\"rocwmma_mfma_packed_b\""
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
