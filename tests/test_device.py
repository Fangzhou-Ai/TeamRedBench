from types import SimpleNamespace

from teamredbench.device import detect_runtime


def test_detect_runtime_reports_rocm_fields_only(monkeypatch):
    monkeypatch.setattr("teamredbench.device._run_command", lambda command: None)

    torch_module = SimpleNamespace(
        __version__="2.6.0",
        version=SimpleNamespace(hip="6.3.0", cuda="12.6"),
    )

    runtime = detect_runtime(torch_module)

    assert runtime["torch"]["torch_version"] == "2.6.0"
    assert runtime["torch"]["hip_version"] == "6.3.0"
    assert "cuda_api_version" not in runtime["torch"]
