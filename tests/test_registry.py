import teamredbench.builtin  # noqa: F401
from teamredbench.registry import list_benchmarks, list_metrics


def test_builtin_registrations_present():
    assert {"hbm", "collective", "mfu"} <= set(list_benchmarks())
    assert {
        "latency_us",
        "hbm_bandwidth_gbps",
        "payload_bandwidth_gbps",
        "bus_bandwidth_gbps",
        "link_efficiency_pct",
        "achieved_tops",
        "mfu_pct",
    } <= set(list_metrics())
