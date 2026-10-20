"""Score the study questionnaires into a CSV (PRD 17.5).

Input CSV columns: participant, condition, sus1..sus10 (1-5), and the six
NASA-TLX ratings tlx_mental, tlx_physical, tlx_temporal, tlx_performance,
tlx_effort, tlx_frustration (0-100).

    python -m tools.questionnaires answers.csv scored.csv
"""

from __future__ import annotations

import argparse

from research.metrics import NASA_TLX_SCALES, nasa_tlx, sus_score


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("out")
    a = ap.parse_args()
    df = pd.read_csv(a.answers)
    df["sus"] = df.apply(lambda r: sus_score([int(r[f"sus{i}"]) for i in range(1, 11)]), axis=1)
    df["tlx_raw"] = df.apply(lambda r: nasa_tlx({k: float(r[f"tlx_{k}"]) for k in NASA_TLX_SCALES}), axis=1)
    df.to_csv(a.out, index=False)
    print(df.groupby("condition")[["sus", "tlx_raw"]].agg(["mean", "std", "count"]).round(1))


if __name__ == "__main__":
    main()
