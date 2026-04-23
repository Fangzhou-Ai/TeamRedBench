#include "mfu_gemm_highp.hip.hpp"
#include "mfu_gemm_lowp.hip.hpp"

int main(int argc, char** argv) {
    using namespace teamredbench::native::mfu_gemm;

    try {
        const Args args = parse_args(argc, argv);
        if(args.dtype == "float16") {
            return emit_low_precision_success<rocwmma::float16_t>(args);
        }
        if(args.dtype == "bfloat16") {
            return emit_low_precision_success<rocwmma::bfloat16_t>(args);
        }
        if(args.dtype == "float32") {
            return emit_high_precision_success<float>(args);
        }
        if(args.dtype == "float64") {
            return emit_high_precision_success<double>(args);
        }
        return emit_skipped("unsupported dtype for native MFU kernel: " + args.dtype);
    } catch(const std::exception& exc) {
        return emit_skipped(exc.what());
    }
}
