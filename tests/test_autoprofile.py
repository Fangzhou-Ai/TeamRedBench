from pathlib import Path

from teamredbench.config import load_suite
from teamredbench.device import DeviceInfo
from teamredbench.runner import build_context


def test_build_context_auto_selects_matching_published_profile(monkeypatch):
    suite = load_suite(Path("configs/suites/full.yaml"))

    monkeypatch.setattr("teamredbench.runner.load_torch", lambda: None)
    monkeypatch.setattr(
        "teamredbench.runner.detect_device",
        lambda torch_module, device_id=0: DeviceInfo(index=device_id, name="AMD Instinct MI350X VF"),
    )

    context, profile_note = build_context(suite, device_id=0)

    assert context.hardware_profile.name == "amd-instinct-mi350x"
    assert profile_note is not None
    assert "MI350X" in profile_note
