"""Shape-family diversity for accepted choreography seeds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed, pairwise_distance_matrix
from fairy_orbit.store.search_db import ChoreographySearchStore, TrialRecord


def shape_feature_vector(seed: OrbitSeed) -> np.ndarray:
    """
    Rotation/scale-ish invariant shape descriptor from IC (COM frame).

    Uses sorted normalized pairwise distances, radius CV / moments, and
    inertia eigenvalue ratios (planarity / elongation).
    """
    r = np.asarray(seed.positions, dtype=float).reshape(-1, 3)
    v = np.asarray(seed.velocities, dtype=float).reshape(-1, 3)
    n = r.shape[0]
    com = np.mean(r, axis=0)
    c = r - com
    radii = np.linalg.norm(c, axis=1)
    r_mean = float(np.mean(radii)) + 1e-300
    radii_n = np.sort(radii / r_mean)

    d = pairwise_distance_matrix(r)
    d_mean = float(np.mean(d)) + 1e-300
    d_n = np.sort(d / d_mean)

    # inertia eigenvalues of point cloud
    I = c.T @ c / max(n, 1)
    w = np.sort(np.linalg.eigvalsh(I))[::-1]
    w = np.maximum(w, 0.0)
    w0 = float(w[0]) + 1e-300
    inert = np.array([w[1] / w0, w[2] / w0], dtype=float)

    # speed structure (scale-free)
    speeds = np.linalg.norm(v, axis=1)
    s_mean = float(np.mean(speeds)) + 1e-300
    speeds_n = np.sort(speeds / s_mean)

    # angular momentum direction vs principal axis (planar vs 3d motion)
    L = np.zeros(3)
    for i in range(n):
        L += np.cross(c[i], v[i])
    L_n = float(np.linalg.norm(L)) + 1e-300
    # alignment of L with least-inertia axis (normal to plane of mass)
    # smallest eigenvalue eigenvector
    _, vecs = np.linalg.eigh(I)
    normal = vecs[:, 0]  # smallest
    align = abs(float(np.dot(L / L_n, normal)))

    period_feat = np.array([float(seed.period) / (2.0 * np.pi * r_mean + 1e-300)])

    return np.concatenate(
        [
            d_n,
            radii_n,
            speeds_n,
            inert,
            np.array([align, float(np.std(radii) / r_mean)]),
            period_feat,
        ]
    ).astype(float)


def shape_distance(fa: np.ndarray, fb: np.ndarray) -> float:
    a = np.asarray(fa, dtype=float)
    b = np.asarray(fb, dtype=float)
    # pad if needed (shouldn't for same n)
    m = min(a.size, b.size)
    return float(np.linalg.norm(a[:m] - b[:m]))


@dataclass(frozen=True)
class ShapeFamilyPick:
    family_id: int
    record: TrialRecord
    seed: OrbitSeed
    feature: np.ndarray
    min_dist_to_prev: float
    residual: float


def select_diverse_families(
    store: ChoreographySearchStore,
    n_bodies: int,
    *,
    n_families: int = 6,
    pool_limit: int | None = None,
    min_sep: float = 0.12,
) -> list[ShapeFamilyPick]:
    """
    Greedy max-min diversity over accepted passes.

    Starts from best residual, then repeatedly adds the candidate that
    maximizes distance to the nearest already-selected family (subject to
    ``min_sep`` when possible).
    """
    records = store.list_passes(n_bodies, limit=pool_limit or 10_000)
    cands: list[tuple[TrialRecord, OrbitSeed, np.ndarray, float]] = []
    for rec in records:
        if rec.seed_json is None or rec.residual is None:
            continue
        try:
            seed = OrbitSeed.from_dict(rec.seed_json)
            feat = shape_feature_vector(seed)
        except Exception:
            continue
        cands.append((rec, seed, feat, float(rec.residual)))
    if not cands:
        return []

    # start with best residual
    cands.sort(key=lambda t: t[3])
    picked: list[ShapeFamilyPick] = []
    first_rec, first_seed, first_feat, first_res = cands[0]
    picked.append(
        ShapeFamilyPick(
            family_id=0,
            record=first_rec,
            seed=first_seed,
            feature=first_feat,
            min_dist_to_prev=0.0,
            residual=first_res,
        )
    )
    used = {first_rec.trial_no}

    while len(picked) < n_families:
        best_i = None
        best_score = -1.0
        best_mind = 0.0
        for rec, seed, feat, res in cands:
            if rec.trial_no in used:
                continue
            mind = min(shape_distance(feat, p.feature) for p in picked)
            # prefer far shapes; light residual tie-break
            score = mind - 0.02 * np.log10(max(res, 1e-30) + 1e-30)
            if score > best_score:
                best_score = score
                best_i = (rec, seed, feat, res)
                best_mind = mind
        if best_i is None:
            break
        # if nothing is far enough and we already have >=3, stop early
        if best_mind < min_sep and len(picked) >= 3:
            # still take if remaining pool has anything reasonably far
            # otherwise stop
            far_left = any(
                min(shape_distance(feat, p.feature) for p in picked) >= min_sep
                for rec, seed, feat, res in cands
                if rec.trial_no not in used
            )
            if not far_left:
                break
            if best_mind < min_sep * 0.5:
                break
        rec, seed, feat, res = best_i
        used.add(rec.trial_no)
        picked.append(
            ShapeFamilyPick(
                family_id=len(picked),
                record=rec,
                seed=seed,
                feature=feat,
                min_dist_to_prev=best_mind,
                residual=res,
            )
        )
    return picked


def families_to_dict(picks: list[ShapeFamilyPick]) -> dict[str, Any]:
    return {
        "n_families": len(picks),
        "families": [
            {
                "family_id": p.family_id,
                "trial_no": p.record.trial_no,
                "residual": p.residual,
                "period": p.seed.period,
                "seed_id": p.seed.id,
                "min_dist_to_prev": p.min_dist_to_prev,
                "result_fp": p.record.result_fp,
            }
            for p in picks
        ],
    }
