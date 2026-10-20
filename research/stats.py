"""Statistical analysis for the user study and experiments (PRD 16.3, 17.6).

Repeated-measures ANOVA (or Friedman when the data are not normal), pairwise
tests with Holm correction, and effect sizes (partial eta squared, Cohen's d).
Input: a long-format table with columns subject, condition, value.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    diff = np.asarray(a, float) - np.asarray(b, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def wide(table: Any, subject: str = "subject", condition: str = "condition", value: str = "value") -> Any:
    return table.pivot_table(index=subject, columns=condition, values=value, aggfunc="mean").dropna()


def repeated_measures(table: Any, subject: str = "subject", condition: str = "condition", value: str = "value",
                      alpha: float = 0.05) -> dict[str, Any]:
    """Omnibus test chosen by a Shapiro-Wilk check on the residuals, then Holm-corrected pairwise tests."""
    w = wide(table, subject, condition, value)
    data = w.to_numpy()
    n, k = data.shape
    resid = (data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + data.mean()).ravel()
    normal = stats.shapiro(resid).pvalue > alpha if len(resid) >= 3 else True
    result: dict[str, Any] = {"n_subjects": n, "conditions": list(w.columns), "normal_residuals": bool(normal)}
    if normal and k >= 2:
        from statsmodels.stats.anova import AnovaRM

        long = table[[subject, condition, value]].groupby([subject, condition], as_index=False).mean()
        long = long[long[subject].isin(w.index)]
        res = AnovaRM(long, value, subject, within=[condition]).fit()
        row = res.anova_table.iloc[0]
        f, df1, df2 = float(row["F Value"]), float(row["Num DF"]), float(row["Den DF"])
        result["omnibus"] = {"test": "rm_anova", "F": f, "df": (df1, df2), "p": float(row["Pr > F"]),
                             "partial_eta_sq": f * df1 / (f * df1 + df2)}
    else:
        chi, p = stats.friedmanchisquare(*[data[:, j] for j in range(k)])
        result["omnibus"] = {"test": "friedman", "chi2": float(chi), "p": float(p),
                             "kendalls_w": float(chi / (n * (k - 1)))}
    pairs, ps = [], []
    for (i, a), (j, b) in itertools.combinations(enumerate(w.columns), 2):
        if normal:
            t = stats.ttest_rel(data[:, i], data[:, j])
            p = float(t.pvalue)
            stat = {"t": float(t.statistic)}
        else:
            t = stats.wilcoxon(data[:, i], data[:, j])
            p = float(t.pvalue)
            stat = {"W": float(t.statistic)}
        pairs.append({"a": a, "b": b, **stat, "p": p, "cohens_d": cohens_d_paired(data[:, i], data[:, j])})
        ps.append(p)
    for pair, padj in zip(pairs, holm(ps), strict=True):
        pair["p_holm"] = padj
        pair["significant"] = padj < alpha
    result["pairwise"] = pairs
    return result


def compare_methods(errors_a: Sequence[float], errors_b: Sequence[float], paired: bool = True) -> dict[str, float]:
    """Per-subject comparison for claims like 'method A beats method B' (p < 0.05)."""
    a, b = np.asarray(errors_a, float), np.asarray(errors_b, float)
    if paired:
        t = stats.ttest_rel(a, b)
        w = stats.wilcoxon(a, b) if len(a) >= 6 else None
        return {"t": float(t.statistic), "p_t": float(t.pvalue), "p_wilcoxon": float(w.pvalue) if w else float("nan"),
                "cohens_d": cohens_d_paired(a, b), "mean_diff": float((a - b).mean())}
    t = stats.ttest_ind(a, b)
    return {"t": float(t.statistic), "p_t": float(t.pvalue), "mean_diff": float(a.mean() - b.mean())}


def mean_sd(values: Sequence[float]) -> str:
    v = np.asarray(values, float)
    return f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.2f}"


def fairness_report(table: Any, group: str, value: str = "error_deg", threshold: float = 0.5) -> dict[str, Any]:
    """Accuracy by group (skin tone, glasses, age; PRD 21.4); gaps over 0.5 degrees are flagged."""
    g = table.groupby(group)[value].agg(["mean", "std", "count"])
    gap = float(g["mean"].max() - g["mean"].min()) if len(g) > 1 else 0.0
    return {"by_group": g.reset_index().to_dict("records"), "gap_deg": gap, "flagged": gap > threshold}
