from __future__ import annotations

from typing import Any

from teamredbench.registry import register_metric


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


@register_metric("latency_us", "Average iteration latency in microseconds.")
def latency_us(raw: dict[str, Any]) -> float | None:
    elapsed = _safe_float(raw.get("elapsed_s"))
    if elapsed is None:
        return None
    return elapsed * 1e6


@register_metric("hbm_bandwidth_gbps", "Device memory bandwidth derived from moved bytes.")
def hbm_bandwidth_gbps(raw: dict[str, Any]) -> float | None:
    elapsed = _safe_float(raw.get("elapsed_s"))
    moved = _safe_float(raw.get("moved_bytes"))
    if elapsed is None or moved is None or elapsed == 0:
        return None
    return moved / elapsed / 1e9


@register_metric("hbm_efficiency_pct", "HBM bandwidth as a percentage of configured peak.")
def hbm_efficiency_pct(raw: dict[str, Any]) -> float | None:
    peak = _safe_float(raw.get("peak_bandwidth_gbps"))
    if peak is None or peak == 0:
        return None
    current = hbm_bandwidth_gbps(raw)
    if current is None:
        return None
    return current / peak * 100.0


@register_metric("payload_bandwidth_gbps", "Collective payload bandwidth in GB/s.")
def payload_bandwidth_gbps(raw: dict[str, Any]) -> float | None:
    elapsed = _safe_float(raw.get("elapsed_s"))
    payload = _safe_float(raw.get("payload_bytes"))
    if elapsed is None or payload is None or elapsed == 0:
        return None
    return payload / elapsed / 1e9


@register_metric("bus_bandwidth_gbps", "Collective link bandwidth in GB/s after bus factor.")
def bus_bandwidth_gbps(raw: dict[str, Any]) -> float | None:
    payload_bw = payload_bandwidth_gbps(raw)
    bus_factor = _safe_float(raw.get("bus_factor"))
    if payload_bw is None or bus_factor is None:
        return None
    return payload_bw * bus_factor


@register_metric("link_efficiency_pct", "Collective bandwidth as a percentage of configured peak link bandwidth.")
def link_efficiency_pct(raw: dict[str, Any]) -> float | None:
    peak = _safe_float(raw.get("peak_link_bandwidth_gbps"))
    current = bus_bandwidth_gbps(raw)
    if peak is None or peak == 0 or current is None:
        return None
    return current / peak * 100.0


@register_metric("achieved_tops", "Achieved throughput in tera-ops per second.")
def achieved_tops(raw: dict[str, Any]) -> float | None:
    elapsed = _safe_float(raw.get("elapsed_s"))
    ops = _safe_float(raw.get("achieved_ops"))
    if elapsed is None or ops is None or elapsed == 0:
        return None
    return ops / elapsed / 1e12


@register_metric("mfu_pct", "Measured throughput as a percentage of configured peak.")
def mfu_pct(raw: dict[str, Any]) -> float | None:
    peak_tops = _safe_float(raw.get("peak_tops"))
    current = achieved_tops(raw)
    if peak_tops is None or peak_tops == 0 or current is None:
        return None
    return current / peak_tops * 100.0
