from __future__ import annotations

import numpy as np


def thomas_cluster_process(
    num_points: int,
    map_size: float,
    num_clusters: int = 3,
    cluster_std: float = 80.0,
    center_min_dist: float = 0.0,
    rng: np.random.Generator | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    num_clusters = max(1, int(num_clusters))
    # Sample cluster centers uniformly, optionally enforcing a minimum separation.
    low = 0.1 * float(map_size)
    high = 0.9 * float(map_size)
    center_min_dist = max(float(center_min_dist or 0.0), 0.0)
    if center_min_dist <= 0.0 or num_clusters <= 1:
        centers = rng.uniform(low, high, size=(num_clusters, 2)).astype(np.float32)
    else:
        best_centers = None
        best_min_dist = -1.0
        for _ in range(512):
            candidate = rng.uniform(low, high, size=(num_clusters, 2)).astype(np.float32)
            diff = candidate[:, None, :] - candidate[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            dist += np.eye(num_clusters, dtype=np.float32) * np.float32(map_size * 10.0 + center_min_dist + 1.0)
            min_dist = float(np.min(dist))
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_centers = candidate
            if min_dist >= center_min_dist:
                break
        centers = best_centers if best_centers is not None else rng.uniform(low, high, size=(num_clusters, 2)).astype(np.float32)
    # Allocate points per cluster
    counts = rng.multinomial(num_points, [1 / num_clusters] * num_clusters).astype(np.int32)
    points = []
    for c, n in zip(centers, counts):
        if n == 0:
            continue
        pts = rng.normal(loc=c, scale=cluster_std, size=(n, 2))
        pts = np.clip(pts, 0.0, map_size)
        points.append(pts)
    if not points:
        pts = rng.uniform(0.0, map_size, size=(num_points, 2)).astype(np.float32)
        if return_metadata:
            return pts, centers, counts
        return pts
    pts = np.vstack(points).astype(np.float32, copy=False)
    if return_metadata:
        return pts, centers, counts
    return pts
