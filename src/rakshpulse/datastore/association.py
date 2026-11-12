"""Associate cuff (reference) BP readings with PPG windows without mislabelling unrelated samples.

A cuff reading describes BP during the cuff measurement, not the whole recording. A window is labelled with a
reading only if all of the following hold:

* the window belongs to the **same subject and session**;
* the reading time lies within ``[t_start − tolerance_s, t_end + tolerance_s]`` of the window (by default the
  reading must fall inside the window or within 30 s of it, about one cuff inflation/deflation cycle);
* the window passed quality gating, if ``require_usable`` and a ``quality_class`` column is present.

When several readings qualify for one window, the closest one is used. No value is ever carried to windows
outside the tolerance (carrying forward is an explicit *method*, see ``personalization.strategies``, never a
labelling step).

Returns (labelled windows, readings with the window(s) they were assigned to or the reason they were not).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def associate_references(
    windows: pd.DataFrame,
    refs: pd.DataFrame,
    tolerance_s: float = 30.0,
    require_usable: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = windows.copy()
    w["ref_sbp"] = np.nan
    w["ref_dbp"] = np.nan
    w["ref_timestamp"] = np.nan
    w["ref_offset_s"] = np.nan
    r = refs.copy().reset_index(drop=True)
    r["n_windows"] = 0
    r["status"] = "unassigned"
    for (sub, ses), g in w.groupby(["subject_id", "session_id"]):
        rr = r[(r.subject_id.astype(str) == str(sub)) & (r.session_id.astype(str) == str(ses))]
        if rr.empty:
            continue
        ok_win = g.quality_class.isin(["good", "degraded"]) if require_usable and "quality_class" in g else None
        for ri, ref in rr.iterrows():
            t = float(ref.timestamp)
            # distance from the reading to the window span (0 if inside)
            dist = np.maximum(g.t_start_s - t, 0) + np.maximum(t - g.t_end_s, 0)
            hit = dist <= tolerance_s
            if ok_win is not None:
                if hit.any() and not (hit & ok_win).any():
                    r.loc[ri, "status"] = "no_usable_window_within_tolerance"
                    continue
                hit &= ok_win
            if not hit.any():
                r.loc[ri, "status"] = "no_window_within_tolerance"
                continue
            idx = g.index[hit]
            off = (t - (g.loc[idx, "t_start_s"] + g.loc[idx, "t_end_s"]) / 2).abs()
            better = w.loc[idx, "ref_offset_s"].isna() | (off < w.loc[idx, "ref_offset_s"].abs())
            take = idx[better.to_numpy()]
            w.loc[take, ["ref_sbp", "ref_dbp", "ref_timestamp"]] = [ref.sbp, ref.dbp, t]
            w.loc[take, "ref_offset_s"] = (t - (w.loc[take, "t_start_s"] + w.loc[take, "t_end_s"]) / 2).to_numpy()
            r.loc[ri, "status"] = "assigned"
    # count final assignments per reading (a window may have switched to a closer reading)
    for ri, ref in r.iterrows():
        n = int(
            (
                (w.subject_id.astype(str) == str(ref.subject_id))
                & (w.session_id.astype(str) == str(ref.session_id))
                & (w.ref_timestamp == float(ref.timestamp))
            ).sum()
        )
        r.loc[ri, "n_windows"] = n
        if ref.status == "assigned" and n == 0:
            r.loc[ri, "status"] = "superseded_by_closer_reading"
    return w, r
