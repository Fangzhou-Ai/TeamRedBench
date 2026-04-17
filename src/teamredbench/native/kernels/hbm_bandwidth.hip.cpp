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
__global__ void scale_kernel(const T* src, T* dst, T alpha, std::size_t count) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for(; index < count; index += stride) {
        dst[index] = src[index] * alpha;
    }
}

template <typename T>
__global__ void triad_kernel(const T* src, T* aux, T* dst, T alpha, std::size_t count) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for(; index < count; index += stride) {
        dst[index] = src[index] + aux[index] * alpha;
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

    const int threads = 256;
    const int max_blocks = std::max(1, props.multiProcessorCount * 8);
    const int blocks = std::min<int>(
        max_blocks,
        static_cast<int>((count + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads))
    );
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
