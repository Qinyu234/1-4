"""Dense early D_Td curve to test log-linear (exponential) growth."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.design.tetra_eff import (
    build_td_system,
    omega_from_vt,
    polar_from_beta_e,
)
from fairy_orbit.engine import ReboundConfig, integrate
from fairy_orbit.observe.calibration import td_breaking

OUT = Path(__file__).resolve().parents[1] / "experiments" / "output" / "td_growth_law"


def main() -> None:
    m, beta, e = 1e-3, 1.0, 0.0
    vr, vt, _ = polar_from_beta_e(m, beta, e)
    omega = omega_from_vt(1.0, vt)
    system, _ = build_td_system(
        m, 1.0, vr, omega, central_radius=0.0, fairy_radius=0.0
    )
    traj = integrate(
        system,
        t_end=0.8,
        n_outputs=800,
        config=ReboundConfig(
            epsilon=1e-9,
            min_dt=1e-5,
            stop_on_escape=False,
            stop_on_collision=False,
        ),
    )
    D = np.array(
        [
            td_breaking(traj.positions[k][1:5] - traj.positions[k][0:1])
            for k in range(len(traj.times))
        ]
    )
    t = traj.times
    i0 = int(np.argmax(D > 1e-15))
    i1 = int(np.argmax(D >= 0.1))
    tt, yy = t[i0:i1], D[i0:i1]
    be = np.polyfit(tt, np.log(yy), 1)
    bl = np.polyfit(tt, yy, 1)
    pred_exp = np.exp(be[0] * tt + be[1])
    pred_lin = bl[0] * tt + bl[1]
    r2log = 1 - np.sum((np.log(yy) - np.log(pred_exp)) ** 2) / np.sum(
        (np.log(yy) - np.log(yy).mean()) ** 2
    )
    ss_log_exp = float(np.sum((np.log(yy) - np.log(pred_exp)) ** 2))
    ss_log_lin = float(np.sum((np.log(yy) - np.log(np.maximum(pred_lin, 1e-30))) ** 2))
    winner = "exp" if ss_log_exp < ss_log_lin else "lin"
    print(
        f"n={len(tt)} t=[{tt[0]:.4f},{tt[-1]:.4f}] D=[{yy[0]:.2e},{yy[-1]:.2e}]"
    )
    print(
        f"lam={be[0]:.3f} decades/t={be[0]/math.log(10):.3f} R2log={r2log:.4f}"
    )
    print(
        f"logSS exp={ss_log_exp:.3e} lin={ss_log_lin:.3e} winner={winner}"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "dense_circular.npz", t=t, D=D)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogy(t, np.maximum(D, 1e-18), "k-", lw=1)
    ax[0].semilogy(tt, pred_exp, "r--", lw=2, label=f"exp λ={be[0]:.1f}")
    ax[0].semilogy(tt, np.maximum(pred_lin, 1e-18), "b:", lw=2, label="lin")
    ax[0].set_xlim(0, 0.8)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$D_{Td}$")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].set_title("dense circular")
    ax[1].plot(tt, np.log10(yy), "k.", ms=3)
    ax[1].plot(
        tt,
        (be[0] * tt + be[1]) / math.log(10),
        "r-",
        label=f"R2log={r2log:.3f}",
    )
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$\log_{10} D_{Td}$")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    ax[1].set_title("log-linear test")
    fig.tight_layout()
    fig.savefig(OUT / "dense_loglinear.png", dpi=140)
    print("saved", OUT)


if __name__ == "__main__":
    main()
