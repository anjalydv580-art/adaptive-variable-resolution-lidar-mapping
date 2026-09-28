"""Shared dashboard data helpers (Phase 15, Streamlit-optional)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
def frame_table(result: Any) -> List[Dict[str, object]]:
    frames = _get(result, "frames", []) or []
    rows: List[Dict[str, object]] = []
    for f in frames:
        feats = _get(f, "features", {}) or {}
        objs = _get(f, "objects", []) or []
        pd = float(feats.get("point_density", 0.0)) if isinstance(
            feats, dict) else 0.0
        rows.append({"frame_id": int(_get(f, "frame_id", 0)),
                     "num_points_raw": int(_get(f, "num_points_raw", 0)),
                     "num_points_processed": int(
                         _get(f, "num_points_processed", 0)),
                     "num_objects": len(objs),
                     "num_dynamic": int(_get(f, "num_dynamic", 0)),
                     "complexity_score": float(
                         _get(f, "complexity_score", 0.0)),
                     "complexity_level": str(_get(f, "complexity_level",
                                                  "LOW")),
                     "ml_prediction": str(_get(f, "ml_prediction", "SIMPLE")),
                     "ml_rule_label": str(_get(f, "ml_rule_label", "SIMPLE")),
                     "point_density": pd})
    return rows
def metric_cards(result: Any) -> Dict[str, float]:
    m = _get(result, "metrics", {}) or {}
    ev = _get(result, "ml_evaluation", {}) or {}
    out: Dict[str, float] = {}
    for k in ("occupied_cell_agreement", "elevation_mae", "elevation_rmse",
              "compression_ratio", "cell_reduction", "speedup"):
        try:
            out[k] = float(m.get(k, float("nan")))
        except (TypeError, ValueError):
            out[k] = float("nan")
    try:
        out["ml_accuracy_vs_rule"] = float(ev.get("accuracy", 0.0))
    except (TypeError, ValueError):
        out["ml_accuracy_vs_rule"] = 0.0
    return out
def _nan_to_none(a: np.ndarray) -> list:
    return [[None if not np.isfinite(v) else float(v) for v in row]
            for row in a]
def elevation_figure_data(result: Any,
                          max_cells: int = 20000) -> Dict[str, object]:
    amap = _get(result, "adaptive_map", None)
    if amap is None:
        return {"elevation": [], "shape": [0, 0],
                "fine_resolution": 0.0, "downsampled": False}
    elev = np.asarray(_get(amap, "elevation", np.zeros((0, 0))),
                      dtype=float)
    fine = float(_get(amap, "fine_resolution", 0.0) or 0.0)
    down = False
    if elev.size > max_cells and elev.size > 0:
        stride = int(max(1, np.ceil(np.sqrt(elev.size / max_cells))))
        elev = elev[::stride, ::stride]
        down = True
    return {"elevation": _nan_to_none(elev),
            "shape": [int(v) for v in elev.shape],
            "fine_resolution": fine, "downsampled": bool(down)}
def region_level_counts(result: Any) -> Dict[str, int]:
    regs = _get(result, "region_decisions", []) or []
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for r in regs:
        lv = str(r.get("resolution_level", "LOW")) if isinstance(
            r, dict) else str(_get(r, "resolution_level", "LOW"))
        if lv in counts:
            counts[lv] += 1
    return counts
def sample_points(result_or_points: Any, max_points: int = 20000,
                  seed: int = 42) -> np.ndarray:
    pts = _get(result_or_points, "points", None)
    arr = np.asarray(pts if pts is not None else result_or_points,
                     dtype=float).reshape(-1, 3)
    if arr.shape[0] <= max_points or arr.shape[0] == 0:
        return arr
    stride = int(max(1, np.ceil(arr.shape[0] / max_points)))
    return arr[::stride]
def build_dashboard_context(result: Any,
                            config: Optional[dict] = None) -> Dict[str, object]:
    title = "Adaptive Variable Resolution 2.5D LiDAR Mapping"
    if isinstance(config, dict):
        title = str(config.get("dashboard", {}).get("title", title))
    return {"title": title,
            "num_frames": int(_get(result, "num_frames", 0)),
            "seed": _get(result, "seed", None),
            "frame_table": frame_table(result),
            "metric_cards": metric_cards(result),
            "elevation": elevation_figure_data(result),
            "region_counts": region_level_counts(result),
            "ml_predictions": [str(v) for v in
                               (_get(result, "ml_predictions", []) or [])],
            "ml_evaluation": dict(_get(result, "ml_evaluation", {}) or {}),
            "ml_model_source": str(_get(result, "ml_model_source",
                                        "unknown")),
            "metrics": dict(_get(result, "metrics", {}) or {})}
def render_dashboard(result: Any,
                     config: Optional[dict] = None) -> None:
    import plotly.express as px
    import streamlit as st
    ctx = build_dashboard_context(result, config)
    st.set_page_config(page_title=str(ctx["title"]), layout="wide")
    st.title(str(ctx["title"]))
    st.caption(f"Frames: {ctx['num_frames']} | Seed: {ctx['seed']} | "
               f"ML model: {ctx['ml_model_source']}")
    cards = ctx["metric_cards"]
    assert isinstance(cards, dict)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cell agreement",
              f"{float(cards.get('occupied_cell_agreement', 0.0)):.3f}")
    c2.metric("Elevation MAE",
              f"{float(cards.get('elevation_mae', 0.0)):.4f} m")
    c3.metric("Cell reduction",
              f"{float(cards.get('cell_reduction', 0.0)):.3f}")
    c4.metric("ML accuracy vs rule",
              f"{float(cards.get('ml_accuracy_vs_rule', 0.0)):.3f}")
    st.subheader("Per-frame summary")
    st.dataframe(ctx["frame_table"])
    st.subheader("Adaptive elevation map (downsampled)")
    elev = ctx["elevation"]
    assert isinstance(elev, dict)
    grid = elev["elevation"]
    arr = np.array([[v if v is not None else np.nan for v in row]
                    for row in grid]) if grid else np.zeros((1, 1))
    fig = px.imshow(arr,
                    title=f"Elevation (fine res "
                          f"{elev['fine_resolution']} m)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Region complexity counts")
    st.bar_chart(ctx["region_counts"])
    st.subheader("ML evaluation vs controller rule")
    st.json(ctx["ml_evaluation"])


