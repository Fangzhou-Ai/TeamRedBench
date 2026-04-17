from pathlib import Path

from teamredbench.config import load_hardware_profile


def test_published_instinct_profiles_load_with_nonzero_peaks():
    hardware_dir = Path("configs/profiles/hardware")
    profile_names = [
        "amd_instinct_mi300x.yaml",
        "amd_instinct_mi325x.yaml",
        "amd_instinct_mi350x.yaml",
        "amd_instinct_mi355x.yaml",
    ]

    for profile_name in profile_names:
        profile = load_hardware_profile(hardware_dir / profile_name)
        assert profile.peak_hbm_bandwidth_gbps and profile.peak_hbm_bandwidth_gbps > 0
        assert profile.peak_link_bandwidth_gbps["xgmi"] > 0
        assert profile.peak_compute_tops["float16"] > 0
        assert profile.peak_compute_tops["bfloat16"] > 0
        assert profile.peak_compute_tops["float32"] > 0
        assert profile.peak_compute_tops["int8"] > 0
