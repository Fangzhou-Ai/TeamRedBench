// Numerical parity check: native MFMA GEMM kernels vs hipBLAS reference.
// Compiled and invoked from tests/test_native_gemm_parity.py.
//
// Usage: gemm_parity <dtype> [M N K]
//   dtype: float16 | bfloat16 | float32 | float64 | all
//   Default shape: 1024 x 1024 x 1024
// Exit code 0 on pass; nonzero if any configured dtype fails its tolerance.

#include "../../src/teamredbench/native/kernels/mfu_gemm_highp.hip.hpp"
#include "../../src/teamredbench/native/kernels/mfu_gemm_lowp.hip.hpp"

#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace teamredbench::native::mfu_gemm;

#define CHECK_HIP(x) do { hipError_t _e = (x); if(_e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(_e)); std::exit(2); } } while(0)
#define CHECK_BLAS(x) do { hipblasStatus_t _e = (x); if(_e != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLAS error %s at %s:%d: %d\n", #x, __FILE__, __LINE__, (int)_e); std::exit(2); } } while(0)

template <typename T>
static T host_cast(float v) { return static_cast<T>(v); }
template <>
rocwmma::bfloat16_t host_cast<rocwmma::bfloat16_t>(float v) { return rocwmma::bfloat16_t(v); }
template <>
rocwmma::float16_t host_cast<rocwmma::float16_t>(float v) { return rocwmma::float16_t(v); }

template <typename T>
static float host_to_float(T v) { return static_cast<float>(v); }
template <>
float host_to_float<rocwmma::bfloat16_t>(rocwmma::bfloat16_t v) { return static_cast<float>(v); }
template <>
float host_to_float<rocwmma::float16_t>(rocwmma::float16_t v) { return static_cast<float>(v); }

template <typename T>
static void fill_random(std::vector<T>& buf, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for(auto& x : buf) x = host_cast<T>(dist(rng));
}

template <typename T>
static void run_native_highp(T* dA, T* dB, T* dC, int M, int N, int K) {
    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    const int tile_blocks = (M / mfma_block_tile_m<T>()) * (N / mfma_block_tile_n<T>());
    const int blocks = std::max(1, std::min(tile_blocks, props.multiProcessorCount * KernelTuning<T>::kMfmaBlocksPerCu));
    hipLaunchKernelGGL(mfma_gemm_highp_kernel<T>, dim3(blocks), dim3(mfma_threads_per_block<T>()),
                       0, 0, dA, dB, dC, M, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

template <typename T>
static void run_native_lowp(T* dA, T* dB, T* dC, int M, int N, int K) {
    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, 0));

    const std::size_t packed_b_count =
        (std::size_t)(K / mfma_block_tile_k<T>())
        * (std::size_t)(N / mfma_block_tile_n<T>())
        * packed_b_tile_elements<T>();
    T* dB_packed = nullptr;
    CHECK_HIP(hipMalloc(&dB_packed, packed_b_count * sizeof(T)));

    const int pack_tiles = (K / mfma_block_tile_k<T>()) * (N / mfma_block_tile_n<T>());
    hipLaunchKernelGGL(pack_b_tiles_kernel<T>, dim3(pack_tiles), dim3(256), 0, 0, dB, dB_packed, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    const int tile_blocks = (M / mfma_block_tile_m<T>()) * (N / mfma_block_tile_n<T>());
    const int blocks = std::max(1, std::min(tile_blocks, props.multiProcessorCount * KernelTuning<T>::kMfmaBlocksPerCu));
    hipLaunchKernelGGL(mfma_gemm_lowp_kernel<T>, dim3(blocks), dim3(mfma_threads_per_block<T>()),
                       0, 0, dA, dB_packed, dC, M, N, K);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipFree(dB_packed));
}

// hipBLAS reference. Native buffers are row-major (A: MxK, B: KxN, C: MxN).
// Row-major C = A*B equals column-major C^T = B^T * A^T with swapped operand
// order and dimensions -- we call gemm(m=N, n=M, k=K) with A,B swapped.
static void ref_sgemm(hipblasHandle_t h, float* dA, float* dB, float* dC, int M, int N, int K) {
    const float a = 1.0f, b = 0.0f;
    CHECK_BLAS(hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &a, dB, N, dA, K, &b, dC, N));
    CHECK_HIP(hipDeviceSynchronize());
}

static void ref_dgemm(hipblasHandle_t h, double* dA, double* dB, double* dC, int M, int N, int K) {
    const double a = 1.0, b = 0.0;
    CHECK_BLAS(hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &a, dB, N, dA, K, &b, dC, N));
    CHECK_HIP(hipDeviceSynchronize());
}

template <typename T>
static void ref_gemm_ex(hipblasHandle_t h, T* dA, T* dB, T* dC, int M, int N, int K, hipDataType dt) {
    const float a = 1.0f, b = 0.0f;
    CHECK_BLAS(hipblasGemmEx(
        h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K,
        &a, dB, dt, N, dA, dt, K,
        &b, dC, dt, N,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT));
    CHECK_HIP(hipDeviceSynchronize());
}

template <typename T>
static int run_case(const std::string& dtype_name, int M, int N, int K, hipDataType blas_dt) {
    std::printf("[%s] M=%d N=%d K=%d\n", dtype_name.c_str(), M, N, K);

    const std::size_t a_count = (std::size_t)M * K;
    const std::size_t b_count = (std::size_t)K * N;
    const std::size_t c_count = (std::size_t)M * N;

    std::vector<T> hA(a_count), hB(b_count);
    const float amp = std::is_same_v<T, rocwmma::float16_t>  ? 0.05f
                    : std::is_same_v<T, rocwmma::bfloat16_t> ? 0.10f
                    : 1.0f;
    fill_random(hA, 0xA1u, -amp, amp);
    fill_random(hB, 0xB2u, -amp, amp);

    T *dA = nullptr, *dB = nullptr, *dC_native = nullptr, *dC_ref = nullptr;
    CHECK_HIP(hipMalloc(&dA, a_count * sizeof(T)));
    CHECK_HIP(hipMalloc(&dB, b_count * sizeof(T)));
    CHECK_HIP(hipMalloc(&dC_native, c_count * sizeof(T)));
    CHECK_HIP(hipMalloc(&dC_ref, c_count * sizeof(T)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), a_count * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), b_count * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dC_native, 0, c_count * sizeof(T)));
    CHECK_HIP(hipMemset(dC_ref, 0, c_count * sizeof(T)));

    if constexpr(std::is_same_v<T, rocwmma::float16_t> || std::is_same_v<T, rocwmma::bfloat16_t>) {
        run_native_lowp<T>(dA, dB, dC_native, M, N, K);
    } else {
        run_native_highp<T>(dA, dB, dC_native, M, N, K);
    }

    hipblasHandle_t handle = nullptr;
    CHECK_BLAS(hipblasCreate(&handle));
    if constexpr(std::is_same_v<T, float>) {
        ref_sgemm(handle, (float*)dA, (float*)dB, (float*)dC_ref, M, N, K);
    } else if constexpr(std::is_same_v<T, double>) {
        ref_dgemm(handle, (double*)dA, (double*)dB, (double*)dC_ref, M, N, K);
    } else {
        ref_gemm_ex<T>(handle, dA, dB, dC_ref, M, N, K, blas_dt);
    }
    CHECK_BLAS(hipblasDestroy(handle));

    std::vector<T> hC_native(c_count), hC_ref(c_count);
    CHECK_HIP(hipMemcpy(hC_native.data(), dC_native, c_count * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(hC_ref.data(), dC_ref, c_count * sizeof(T), hipMemcpyDeviceToHost));

    double max_abs = 0.0, ref_max = 0.0, sum_abs = 0.0;
    for(std::size_t i = 0; i < c_count; ++i) {
        const double n = static_cast<double>(host_to_float<T>(hC_native[i]));
        const double r = static_cast<double>(host_to_float<T>(hC_ref[i]));
        const double diff = std::fabs(n - r);
        max_abs = std::max(max_abs, diff);
        ref_max = std::max(ref_max, std::fabs(r));
        sum_abs += diff;
    }
    const double mean_abs = sum_abs / static_cast<double>(c_count);

    const double tol_rel = std::is_same_v<T, double>               ? 1e-10
                         : std::is_same_v<T, float>                ? 1e-4
                         : std::is_same_v<T, rocwmma::bfloat16_t>  ? 2e-2
                         :                                           1e-2;  // fp16
    const bool pass = !std::isnan(mean_abs) && (max_abs <= tol_rel * std::max(ref_max, 1e-12));

    std::printf("  ref |max|       = %.6e\n", ref_max);
    std::printf("  max abs error  = %.6e\n", max_abs);
    std::printf("  mean abs error = %.6e\n", mean_abs);
    std::printf("  tol (rel)      = %.1e   =>  %s\n", tol_rel, pass ? "PASS" : "FAIL");

    CHECK_HIP(hipFree(dC_ref));
    CHECK_HIP(hipFree(dC_native));
    CHECK_HIP(hipFree(dB));
    CHECK_HIP(hipFree(dA));
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    const std::string only = (argc >= 2) ? argv[1] : "all";
    const int M = (argc >= 5) ? std::atoi(argv[2]) : 1024;
    const int N = (argc >= 5) ? std::atoi(argv[3]) : 1024;
    const int K = (argc >= 5) ? std::atoi(argv[4]) : 1024;

    int fails = 0;
    if(only == "all" || only == "float16")
        fails += run_case<rocwmma::float16_t>("float16", M, N, K, HIP_R_16F);
    if(only == "all" || only == "bfloat16")
        fails += run_case<rocwmma::bfloat16_t>("bfloat16", M, N, K, HIP_R_16BF);
    if(only == "all" || only == "float32")
        fails += run_case<float>("float32", M, N, K, HIP_R_32F);
    if(only == "all" || only == "float64")
        fails += run_case<double>("float64", M, N, K, HIP_R_64F);

    std::printf("%s\n", fails == 0 ? "ALL PASS" : "SOME FAILED");
    return fails == 0 ? 0 : 1;
}
