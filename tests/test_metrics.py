from teamredbench.metrics.common import (
    achieved_tops,
    bus_bandwidth_gbps,
    hbm_bandwidth_gbps,
    hbm_efficiency_pct,
    latency_us,
    link_efficiency_pct,
    mfu_pct,
    payload_bandwidth_gbps,
)


def test_hbm_metrics():
    raw = {
        "elapsed_s": 0.5,
        "moved_bytes": 1_000_000_000,
        "peak_bandwidth_gbps": 4.0,
    }
    assert hbm_bandwidth_gbps(raw) == 2.0
    assert hbm_efficiency_pct(raw) == 50.0
    assert latency_us(raw) == 500000.0


def test_collective_metrics():
    raw = {
        "elapsed_s": 0.25,
        "payload_bytes": 500_000_000,
        "bus_factor": 1.5,
        "peak_link_bandwidth_gbps": 4.0,
    }
    assert payload_bandwidth_gbps(raw) == 2.0
    assert bus_bandwidth_gbps(raw) == 3.0
    assert link_efficiency_pct(raw) == 75.0


def test_mfu_metrics():
    raw = {
        "elapsed_s": 0.1,
        "achieved_ops": 5_000_000_000_000,
        "peak_tops": 100.0,
    }
    assert achieved_tops(raw) == 50.0
    assert mfu_pct(raw) == 50.0
