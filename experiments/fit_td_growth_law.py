"""Re-diagnose Td error growth law with proper early window."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output" / "td_growth_law"


def early_mask(t, y, y_hi=0.1):
    """From first y>1e-15 up to first y>=y_hi (exclusive of saturation)."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    good = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if not np.any(good & (y > 1e-15)):
        return np.zeros_like(y, dtype=bool)
    i0 = int(np.argmax(good & (y > 1e-15)))
    above = np.where(good & (np.arange(len(y)) >= i0) & (y >= y_hi))[0]
    i1 = int(above[0]) if len(above) else int(np.where(good)[0][-1] + 1)
    m = np.zeros_like(y, dtype=bool)
    m[i0:i1] = good[i0:i1] & (y[i0:i1] > 1e-16)
    return m


def fit_loglin(t, y):
    """Return λ, R² of log y = c + λ t, and linear R² of y=a+bt."""
    if len(t) < 4:
        return None
    lz = np.log(y)
    # exp in log space
    be = np.polyfit(t, lz, 1)
    pred_log = be[0] * t + be[1]
    ss_log = float(np.sum((lz - pred_log) ** 2))
    ss_log_tot = float(np.sum((lz - lz.mean()) ** 2)) + 1e-30
    r2_log = 1.0 - ss_log / ss_log_tot
    # linear in lin space
    bl = np.polyfit(t, y, 1)
    pred_lin = bl[0] * t + bl[1]
    ss_lin = float(np.sum((y - pred_lin) ** 2))
    ss_lin_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-30
    r2_lin = 1.0 - ss_lin / ss_lin_tot
    # also R² of log under a linear model (bad if true exp)
    # compare AIC-like: residual in log space for both
    # for linear model mapped to log: log|pred_lin|
    pred_lin_pos = np.maximum(pred_lin, 1e-30)
    ss_lin_as_log = float(np.sum((lz - np.log(pred_lin_pos)) ** 2))
    return {
        "n": len(t),
        "t0": float(t[0]),
        "t1": float(t[-1]),
        "lam": float(be[0]),
        "decades": float(be[0] / np.log(10)),
        "r2_logexp": float(r2_log),
        "r2_lin": float(r2_lin),
        "ss_log_exp": ss_log,
        "ss_log_linmodel": ss_lin_as_log,
        "winner_logSS": "exp" if ss_log <= ss_lin_as_log else "lin",
        "slope_lin": float(bl[0]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = list((ROOT / "experiments/output/td_beta_e_scan").glob("b*_e*.npz"))
    files += list((ROOT / "experiments/output/td_error_growth").glob("*/series.npz"))

    rows = []
    for p in files:
        d = np.load(p)
        t = d["t"]
        for qty, key in (("D_Td", "D_Td"), ("B", "err_B"), ("A", "err_A")):
            if key not in d.files:
                continue
            y = d[key]
            m = early_mask(t, y, y_hi=0.1)
            if int(m.sum()) < 4:
                # try softer ceiling
                m = early_mask(t, y, y_hi=1.0)
            if int(m.sum()) < 4:
                continue
            r = fit_loglin(t[m], y[m])
            if r is None:
                continue
            rows.append({"file": p.stem if p.suffix == ".npz" else p.parent.name, "qty": qty, **r})

    print(f"n fits={len(rows)}")
    for qty in ("D_Td", "B", "A"):
        sub = [r for r in rows if r["qty"] == qty]
        if not sub:
            continue
        n_exp = sum(1 for r in sub if r["winner_logSS"] == "exp")
        print(f"\n=== {qty} n={len(sub)} exp_wins(logSS)={n_exp}/{len(sub)} ===")
        print(
            f"  λ median={np.median([r['lam'] for r in sub]):.2f}  "
            f"decades/t={np.median([r['decades'] for r in sub]):.2f}"
        )
        print(
            f"  R²(log-exp)={np.median([r['r2_logexp'] for r in sub]):.3f}  "
            f"R²(lin)={np.median([r['r2_lin'] for r in sub]):.3f}"
        )
        print(
            f"  window pts median={np.median([r['n'] for r in sub]):.0f}  "
            f"t1 median={np.median([r['t1'] for r in sub]):.3f}"
        )
        for r in sorted(sub, key=lambda x: -x["r2_logexp"])[:3]:
            print(
                f"  {r['file']}: λ={r['lam']:.2f} R2log={r['r2_logexp']:.3f} "
                f"R2lin={r['r2_lin']:.3f} win={r['winner_logSS']} n={r['n']}"
            )

    # plot circular reference
    p = ROOT / "experiments/output/td_error_growth/m1e-03_b1.00_e0.00/series.npz"
    d = np.load(p)
    t, D, B = d["t"], d["D_Td"], d["err_B"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, y, title in (
        (axes[0, 0], D, r"$D_{Td}$"),
        (axes[0, 1], B, "B posRMS"),
    ):
        m = early_mask(t, y, 0.1)
        r = fit_loglin(t[m], y[m])
        ax.semilogy(t, np.maximum(y, 1e-18), "k-", lw=1, alpha=0.7, label="data")
        if r is not None:
            tt = t[m]
            pred = np.exp(r["lam"] * (tt - tt[0]) + np.log(y[m][0]))
            # use actual polyfit pred
            be = np.polyfit(tt, np.log(y[m]), 1)
            pred = np.exp(be[0] * tt + be[1])
            ax.semilogy(
                tt,
                pred,
                "r--",
                lw=2,
                label=rf"exp λ={r['lam']:.1f}, R²log={r['r2_logexp']:.2f}",
            )
            bl = np.polyfit(tt, y[m], 1)
            ax.semilogy(
                tt,
                np.maximum(bl[0] * tt + bl[1], 1e-18),
                "b:",
                lw=2,
                label=rf"lin R²={r['r2_lin']:.2f}",
            )
        ax.axhline(0.1, color="gray", ls=":", lw=0.8)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("t")
        ax.set_ylabel(title)
        ax.set_title(title + " early growth")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    # log10 vs t — should be straight if exp
    m = early_mask(t, D, 0.1)
    axes[1, 0].plot(t[m], np.log10(D[m]), "ko", ms=4, label="data")
    be = np.polyfit(t[m], np.log(D[m]), 1)
    axes[1, 0].plot(
        t[m],
        (be[0] * t[m] + be[1]) / np.log(10),
        "r-",
        label=f"slope={be[0]/np.log(10):.2f} dex/t",
    )
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel(r"$\log_{10} D_{Td}$")
    axes[1, 0].set_title("log-linear test")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # histogram of R2_logexp - R2_lin for D_Td
    sub = [r for r in rows if r["qty"] == "D_Td"]
    diff = [r["r2_logexp"] - r["r2_lin"] for r in sub]
    axes[1, 1].hist(diff, bins=10, color="C0", edgecolor="k")
    axes[1, 1].axvline(0, color="r", ls="--")
    axes[1, 1].set_xlabel(r"$R^2_{log-exp} - R^2_{lin}$")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("D_Td: >0 favors exp (in R² sense)")

    fig.suptitle("Td early error growth law")
    fig.tight_layout()
    fig.savefig(OUT / "growth_law.png", dpi=140)
    plt.close(fig)

    dtd = [r for r in rows if r["qty"] == "D_Td"]
    bb = [r for r in rows if r["qty"] == "B"]
    lines = [
        "# Td error vs time: linear or exponential?",
        "",
        "Early window: first rise until `err` hits 0.1 (pre-saturation).",
        "Judge in log space (fair when error spans many decades).",
        "",
        f"## D_Td ({len(dtd)} cases)",
        f"- exp wins (log SS): {sum(1 for r in dtd if r['winner_logSS']=='exp')}/{len(dtd)}",
        f"- median λ ≈ {np.median([r['lam'] for r in dtd]):.2f}  →  err ~ e^{{λt}}",
        f"- median {np.median([r['decades'] for r in dtd]):.2f} decades per unit time",
        f"- median R²(log-exp)={np.median([r['r2_logexp'] for r in dtd]):.3f}, "
        f"R²(lin)={np.median([r['r2_lin'] for r in dtd]):.3f}",
        "",
        f"## B model error ({len(bb)} cases)",
        f"- exp wins: {sum(1 for r in bb if r['winner_logSS']=='exp')}/{len(bb)}",
        f"- median λ ≈ {np.median([r['lam'] for r in bb]):.2f}",
        f"- median R²(log-exp)={np.median([r['r2_logexp'] for r in bb]):.3f}, "
        f"R²(lin)={np.median([r['r2_lin'] for r in bb]):.3f}",
        "",
        "After ~t≳1, D_Td saturates at O(1) and oscillates — growth law no longer applies.",
        "",
        "Plot: `growth_law.png`.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
