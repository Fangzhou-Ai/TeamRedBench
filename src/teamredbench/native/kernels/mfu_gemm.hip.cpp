#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string dtype = "float32";
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 10;
    int iterations = 50;
    int device_id = 0;
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

void check_hipblas(hipblasStatus_t status, const char* operation) {
    if(status != HIPBLAS_STATUS_SUCCESS) {
        std::ostringstream buffer;
        buffer << operation << " failed: " << hipblasStatusToString(status);
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
        } else {
            fail("unknown argument " + key);
        }
    }
    return args;
}

template <typename T>
T* device_alloc(std::size_t count) {
    T* pointer = nullptr;
    check_hip(hipMalloc(&reinterpret_cast<void*&>(pointer), count * sizeof(T)), "hipMalloc");
    check_hip(hipMemset(pointer, 0, count * sizeof(T)), "hipMemset");
    return pointer;
}

template <typename Scalar>
double run_typed_gemm(const Args& args, hipDataType dtype, hipblasComputeType_t compute_type) {
    check_hip(hipSetDevice(args.device_id), "hipSetDevice");

    const std::size_t a_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.k);
    const std::size_t b_count = static_cast<std::size_t>(args.k) * static_cast<std::size_t>(args.n);
    const std::size_t c_count = static_cast<std::size_t>(args.m) * static_cast<std::size_t>(args.n);

    Scalar* a = device_alloc<Scalar>(a_count);
    Scalar* b = device_alloc<Scalar>(b_count);
    Scalar* c = device_alloc<Scalar>(c_count);

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    hipblasHandle_t handle = nullptr;

    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
    check_hipblas(hipblasCreate(&handle), "hipblasCreate");
    check_hipblas(hipblasSetStream(handle, stream), "hipblasSetStream");

    const float alpha_fp32 = 1.0f;
    const float beta_fp32 = 0.0f;
    const double alpha_fp64 = 1.0;
    const double beta_fp64 = 0.0;
    const void* alpha = compute_type == HIPBLAS_COMPUTE_64F ? static_cast<const void*>(&alpha_fp64)
                                                            : static_cast<const void*>(&alpha_fp32);
    const void* beta = compute_type == HIPBLAS_COMPUTE_64F ? static_cast<const void*>(&beta_fp64)
                                                           : static_cast<const void*>(&beta_fp32);

    auto run_once = [&]() {
        check_hipblas(
            hipblasGemmEx(
                handle,
                HIPBLAS_OP_N,
                HIPBLAS_OP_N,
                args.m,
                args.n,
                args.k,
                alpha,
                a,
                dtype,
                args.m,
                b,
                dtype,
                args.k,
                beta,
                c,
                dtype,
                args.m,
                compute_type,
                HIPBLAS_GEMM_DEFAULT
            ),
            "hipblasGemmEx"
        );
    };

    for(int iteration = 0; iteration < args.warmup; ++iteration) {
        run_once();
    }
    check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize(warmup)");

    check_hip(hipEventRecord(start, stream), "hipEventRecord(start)");
    for(int iteration = 0; iteration < args.iterations; ++iteration) {
        run_once();
    }
    check_hip(hipEventRecord(stop, stream), "hipEventRecord(stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");

    if(handle != nullptr) {
        hipblasDestroy(handle);
    }
    if(stop != nullptr) {
        hipEventDestroy(stop);
    }
    if(start != nullptr) {
        hipEventDestroy(start);
    }
    if(stream != nullptr) {
        hipStreamDestroy(stream);
    }
    if(c != nullptr) {
        hipFree(c);
    }
    if(b != nullptr) {
        hipFree(b);
    }
    if(a != nullptr) {
        hipFree(a);
    }

    const int iters = std::max(args.iterations, 1);
    return static_cast<double>(elapsed_ms) / 1000.0 / static_cast<double>(iters);
}

template <typename Scalar>
int emit_success(const Args& args, hipDataType dtype, hipblasComputeType_t compute_type) {
    const double elapsed_s = run_typed_gemm<Scalar>(args, dtype, compute_type);
    std::cout << "{\"status\":\"ok\",\"raw_metrics\":{\"elapsed_s\":" << elapsed_s
              << "},\"metadata\":{\"implementation\":\"hipblas\"}}" << std::endl;
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
            return emit_success<__half>(args, HIPBLAS_R_16F, HIPBLAS_COMPUTE_32F);
        }
        if(args.dtype == "bfloat16") {
            return emit_success<hip_bfloat16>(args, HIPBLAS_R_16B, HIPBLAS_COMPUTE_32F);
        }
        if(args.dtype == "float32") {
            return emit_success<float>(args, HIPBLAS_R_32F, HIPBLAS_COMPUTE_32F);
        }
        if(args.dtype == "float64") {
            return emit_success<double>(args, HIPBLAS_R_64F, HIPBLAS_COMPUTE_64F);
        }
        return emit_skipped("unsupported dtype for native MFU kernel: " + args.dtype);
    } catch(const std::exception& exc) {
        return emit_skipped(exc.what());
    }
}
