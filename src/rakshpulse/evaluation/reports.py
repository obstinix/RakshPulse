"""Turn experiment results into Markdown tables."""

from __future__ import annotations

import pandas as pd


def results_table(rows: list[dict], cols: list[str] | None = None) -> str:
    df = pd.DataFrame(rows)
    cols = cols or list(df.columns)
    df = df[cols]
    fmt = df.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in fmt.to_numpy()]
    return "\n".join([header, sep, *body])
