"""Model-bundle contract: what a model needs, what it is suitable for, and strict input validation.

Every inference path (``inference.engine``, ``inference.predictor``, ``rakshpulse analyze``) builds its model input
through :func:`build_feature_vector` after :func:`check_compatibility`. Nothing reaches a model without passing
these checks, and nothing is imputed silently.

A contract (``ModelBundle.contract``, a plain dict saved inside the joblib bundle) declares:

* ``model_id`` and ``contract_version``;
* ``features``: an ordered list (identical to ``bundle.features``). Each entry has ``name`` (model input name),
  ``source`` (the pipeline feature it is read from, e.g. ``ir_hr`` for ``ear_hr``), ``kind`` (``signal`` or
  ``demographic``), ``required`` and ``fill``:
  - **required** features must be present and finite in every window, otherwise the window is rejected;
  - **imputable** features may be missing in a window. They are filled with ``fill``, the exact median the model
    was trained with. At most ``max_imputed_fraction`` of the signal features may be filled, and the filled names
    are always reported with the prediction;
* ``demographics``: required demographic inputs with plausible ranges (never imputed);
* ``channels``: pipeline channels the features come from (``ir``, ``red``);
* ``fs_range_hz``: accepted *raw* sampling rates. ``processing`` holds the preprocessing that produced the training
  features (target rate, window length, band-pass, feature-set version). A different pipeline configuration is
  refused;
* ``validated_for`` / ``experimental_for`` / ``demonstration_for``: input sensors and the mode a prediction runs
  in. ``validated`` is reserved for a model independently validated on that sensor (no current model).
  ``experimental`` means trained on that sensor type and only cross-validated. ``demonstration`` means the output
  only demonstrates the pipeline and **is not a BP measurement**. Any other sensor is unsupported and refused;
* ``uncertainty``: ``{"method": ..., "halfwidth_90": {"sbp": .., "dbp": ..}, "valid_for": [sensors]}``. Intervals
  are reported only for sensors in ``valid_for``. Otherwise uncertainty is reported as unavailable;
* ``calibration``: whether cuff calibration is supported;
* ``performance`` and ``limitations``: free text and numbers shown by ``rakshpulse models``.

Bundles saved before contracts existed get a conservative *legacy* contract at load time (demonstration only,
no uncertainty, the core features required).
"""

from __future__ import annotations

import numpy as np

CONTRACT_VERSION = 1
FEATURE_SET_VERSION = "rakshpulse-features-1"  # 1.0 per-channel features + normalised shape + 4.0 contour features
DEFAULT_PROCESSING = {
    "target_fs_hz": 100.0,
    "window_s": 10.0,
    "band_hz": [0.5, 10.0],
    "feature_set": FEATURE_SET_VERSION,
}
DEMOGRAPHIC_RANGES = {"age": (10.0, 110.0), "sex": (0.0, 1.0), "height_cm": (100.0, 230.0), "weight_kg": (25.0, 250.0)}
CORE_REQUIRED = ("hr", "beat_T", "sys_ratio", "w50")  # always produced when a window has beats
MODES = ("validated", "experimental", "demonstration")


class IncompatibleModelError(ValueError):
    """The bundle cannot be used with this input (sensor, channels, sampling rate, preprocessing, demographics)."""


class InvalidInputError(ValueError):
    """A specific window/input cannot be turned into a valid model input (missing/non-finite required values)."""


def _strip_site(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def legacy_contract(bundle, prefix_map: dict | None = None) -> dict:
    """Conservative contract for bundles saved without one: demonstration only, no uncertainty."""
    prefix_map = prefix_map or {}
    inv = {v: k for k, v in prefix_map.items()}
    feats = []
    imp = bundle.sbp.named_steps.get("impute") if hasattr(bundle.sbp, "named_steps") else None
    fills = list(getattr(imp, "statistics_", [np.nan] * len(bundle.features)))
    for name, fill in zip(bundle.features, fills, strict=False):
        demo = name.startswith("demo_")
        src = name
        for model_pre, pipe_pre in inv.items():
            if name.startswith(model_pre):
                src = pipe_pre + name[len(model_pre) :]
        feats.append(
            {
                "name": name,
                "source": src,
                "kind": "demographic" if demo else "signal",
                "required": demo or _strip_site(name) in CORE_REQUIRED,
                "fill": None if demo else float(fill),
            }
        )
    chans = sorted({f["source"].split("_")[0] for f in feats if f["kind"] == "signal"} & {"ir", "red"}) or ["ir"]
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": f"legacy:{bundle.version}",
        "legacy": True,
        "features": feats,
        "channels": chans,
        "demographics": [f["name"][5:] for f in feats if f["kind"] == "demographic"],
        "fs_range_hz": [25.0, 1000.0],
        "processing": dict(DEFAULT_PROCESSING),
        "max_imputed_fraction": 0.25,
        "validated_for": [],
        "experimental_for": [],
        "demonstration_for": ["*"],
        "uncertainty": {"method": "none", "halfwidth_90": {}, "valid_for": []},
        "calibration": {"supported": True},
        "performance": {},
        "limitations": ["bundle saved without a contract: demonstration use only"],
    }


def get_contract(bundle, prefix_map: dict | None = None) -> dict:
    c = getattr(bundle, "contract", None)
    return c if c else legacy_contract(bundle, prefix_map)


def validate_bundle(bundle, prefix_map: dict | None = None) -> dict:
    """Structural checks of a bundle and its contract. Returns the contract; raises IncompatibleModelError."""
    errs = []
    for t in ("sbp", "dbp"):
        if not hasattr(getattr(bundle, t, None), "predict"):
            errs.append(f"bundle has no fitted '{t}' model")
    if len(set(bundle.features)) != len(bundle.features):
        errs.append("duplicate feature names in bundle.features")
    c = get_contract(bundle, prefix_map)
    if c.get("contract_version", 0) > CONTRACT_VERSION:
        errs.append(f"contract version {c.get('contract_version')} is newer than this software ({CONTRACT_VERSION})")
    names = [f["name"] for f in c.get("features", [])]
    if names != list(bundle.features):
        errs.append("contract feature list does not match bundle.features (name or order)")
    for t in ("sbp", "dbp"):
        n_in = getattr(getattr(bundle, t, None), "n_features_in_", None)
        if n_in is not None and n_in != len(bundle.features):
            errs.append(f"{t} model expects {n_in} inputs but the bundle lists {len(bundle.features)} features")
    for f in c.get("features", []):
        if f["kind"] == "signal" and not f["required"] and (f.get("fill") is None or not np.isfinite(f["fill"])):
            errs.append(f"imputable feature {f['name']} has no finite training fill value")
    if errs:
        raise IncompatibleModelError("; ".join(errs))
    return c


def inference_mode(contract: dict, sensor: str) -> str | None:
    for mode, key in (
        ("validated", "validated_for"),
        ("experimental", "experimental_for"),
        ("demonstration", "demonstration_for"),
    ):
        lst = contract.get(key, [])
        if sensor in lst or "*" in lst:
            return mode
    return None


def check_demographics(contract: dict, demographics: dict | None) -> dict[str, float]:
    need = contract.get("demographics", [])
    if not need:
        return {}
    demographics = demographics or {}
    missing = [k for k in need if k not in demographics or demographics[k] is None]
    if missing:
        raise IncompatibleModelError(f"this model requires demographic inputs {need}; missing {missing}")
    out = {}
    for k in need:
        try:
            v = float(demographics[k])
        except (TypeError, ValueError) as exc:
            raise IncompatibleModelError(f"demographic '{k}' must be numeric, got {demographics[k]!r}") from exc
        lo, hi = DEMOGRAPHIC_RANGES.get(k, (-np.inf, np.inf))
        if not np.isfinite(v) or not lo <= v <= hi:
            raise IncompatibleModelError(f"demographic '{k}'={v} outside the plausible range [{lo}, {hi}]")
        out[k] = v
    return out


def check_compatibility(
    contract: dict,
    sensor: str,
    fs: float,
    processing: dict | None = None,
    channels: tuple[str, ...] = ("ir", "red"),
    demographics: dict | None = None,
) -> dict:
    """Decide whether this model may run on this input. Returns {"mode", "demographics", "uncertainty"} or raises."""
    mode = inference_mode(contract, sensor)
    if mode is None:
        allowed = (
            contract.get("validated_for", [])
            + contract.get("experimental_for", [])
            + contract.get("demonstration_for", [])
        )
        raise IncompatibleModelError(
            f"model {contract.get('model_id')} does not support sensor '{sensor}' (supported: {allowed or 'none'})"
        )
    lo, hi = contract.get("fs_range_hz", [25.0, 1000.0])
    if not (lo <= float(fs) <= hi):
        raise IncompatibleModelError(f"sampling rate {fs} Hz outside the model's supported range [{lo}, {hi}] Hz")
    need = set(contract.get("channels", []))
    if not need <= set(channels):
        raise IncompatibleModelError(f"model needs channels {sorted(need)}; input provides {sorted(channels)}")
    want = contract.get("processing", DEFAULT_PROCESSING)
    have = {**DEFAULT_PROCESSING, **(processing or {})}
    for k in ("target_fs_hz", "window_s", "band_hz", "feature_set"):
        a, b = want.get(k), have.get(k)
        same = np.allclose(a, b) if isinstance(a, (int, float, list, tuple)) and not isinstance(a, str) else a == b
        if not same:
            raise IncompatibleModelError(f"preprocessing mismatch for '{k}': model trained with {a}, input uses {b}")
    demo = check_demographics(contract, demographics)
    unc = contract.get("uncertainty", {})
    unc_ok = unc.get("method", "none") != "none" and sensor in unc.get("valid_for", [])
    return {
        "mode": mode,
        "demographics": demo,
        "uncertainty": {t: float(v) for t, v in unc.get("halfwidth_90", {}).items()} if unc_ok else {},
        "uncertainty_note": ""
        if unc_ok
        else (
            "uncertainty unavailable: the model has no interval estimate"
            if unc.get("method", "none") == "none"
            else f"uncertainty unavailable for sensor '{sensor}' (interval estimated on "
            f"{unc.get('valid_for')}; not transferable)"
        ),
    }


def build_feature_vector(
    contract: dict, features: dict, demographics: dict[str, float]
) -> tuple[np.ndarray, list[str]]:
    """Ordered 1×n model input from pipeline features. Returns (X, imputed feature names) or raises InvalidInputError.

    Inf is never accepted. Required features must be finite. Imputable features are filled with the training median,
    and too many fills reject the window."""
    x, imputed, bad = [], [], []
    n_signal = 0
    for f in contract["features"]:
        if f["kind"] == "demographic":
            x.append(float(demographics[f["name"][5:] if f["name"].startswith("demo_") else f["name"]]))
            continue
        n_signal += 1
        v = features.get(f["source"], np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise InvalidInputError(f"feature {f['source']} is not numeric: {v!r}") from None
        if np.isinf(v):
            raise InvalidInputError(f"feature {f['source']} is infinite")
        if not np.isfinite(v):
            if f["required"]:
                bad.append(f["source"])
                continue
            imputed.append(f["name"])
            v = float(f["fill"])
        x.append(v)
    if bad:
        raise InvalidInputError(f"required features missing or non-finite: {bad}")
    frac = len(imputed) / max(n_signal, 1)
    if frac > contract.get("max_imputed_fraction", 0.25):
        raise InvalidInputError(
            f"{len(imputed)}/{n_signal} features missing (> {contract.get('max_imputed_fraction')} "
            f"allowed): {imputed[:6]}"
        )
    X = np.asarray(x, float)[None, :]
    if not np.isfinite(X).all():  # defensive: nothing non-finite may reach a model
        raise InvalidInputError("non-finite model input after validation")
    return X, imputed


def summary(contract: dict) -> dict:
    """Flat human-readable summary (used by ``rakshpulse models``)."""
    f = contract["features"]
    return {
        "model_id": contract.get("model_id"),
        "legacy_contract": bool(contract.get("legacy")),
        "channels": ",".join(contract.get("channels", [])),
        "features": len(f),
        "required_features": sum(1 for x in f if x["required"] and x["kind"] == "signal"),
        "imputable_features": sum(1 for x in f if not x["required"] and x["kind"] == "signal"),
        "demographics": ",".join(contract.get("demographics", [])) or "none",
        "fs_range_hz": contract.get("fs_range_hz"),
        "validated_for": ",".join(contract.get("validated_for", [])) or "none",
        "experimental_for": ",".join(contract.get("experimental_for", [])) or "none",
        "demonstration_for": ",".join(contract.get("demonstration_for", [])) or "none",
        "uncertainty": f"{contract.get('uncertainty', {}).get('method', 'none')} "
        f"(valid for: {','.join(contract.get('uncertainty', {}).get('valid_for', [])) or 'none'})",
        "limitations": " | ".join(contract.get("limitations", [])),
    }
