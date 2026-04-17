from pathlib import Path

from teamredbench.config import load_suite


def test_load_suite_resolves_profiles():
    suite = load_suite(Path("configs/suites/smoke.yaml"))
    assert suite.name == "smoke"
    assert suite.hardware_profile.family == "cdna"
    assert suite.runtime_profile.rocm_version == "6.x"
    assert suite.outputs.directory.name == "smoke"

