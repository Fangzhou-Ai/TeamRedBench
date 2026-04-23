from pathlib import Path

from teamredbench.config import load_suite


def test_load_suite_resolves_profiles():
    suite = load_suite(Path("configs/suites/smoke.yaml"))
    assert suite.name == "smoke"
    assert suite.hardware_profile.family == "cdna"
    assert suite.runtime_profile.rocm_version == "6.x"
    assert suite.outputs.directory.name == "smoke"
    assert suite.profiling.enabled is False


def test_load_gemm_only_smoke_suites():
    for path, suite_name, output_name, backend in [
        ("configs/suites/gemm_smoke.yaml", "gemm-smoke", "gemm_smoke", "torch"),
        ("configs/suites/native_gemm_smoke.yaml", "native-gemm-smoke", "native_gemm_smoke", "native"),
    ]:
        suite = load_suite(Path(path))
        assert suite.name == suite_name
        assert suite.outputs.directory.name == output_name
        assert len(suite.benchmarks) == 1

        benchmark = suite.benchmarks[0]
        assert benchmark.benchmark == "mfu"
        assert benchmark.params["backend"] == backend
        assert benchmark.params["dtypes"] == ["float16", "bfloat16", "float32", "float64"]
