from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional

import numpy as np

RC = Tuple[int, int]


def xy_to_rc(x_m: float, y_m: float, *, nr: int, scale: float) -> RC:
    c = int(round(x_m / scale))
    r = int(nr - round(y_m / scale))
    return r, c


def rc_to_xy(r: int, c: int, *, nr: int, scale: float) -> Tuple[float, float]:
    x_m = float(c) * float(scale)
    y_m = float(nr - r) * float(scale)
    return x_m, y_m


def objective_function(height_map: np.ndarray, rc: RC) -> float:
    r, c = rc
    return float(height_map[r, c])


def get_valid_neighbors(height_map: np.ndarray, rc: RC, max_step: float) -> List[RC]:
    nr, nc = height_map.shape
    r, c = rc
    cur_h = float(height_map[r, c])

    neighbors: List[RC] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if not (0 <= rr < nr and 0 <= cc < nc):
                continue

            nh = float(height_map[rr, cc])
            if nh < 0:
                continue

            if abs(nh - cur_h) > max_step:
                continue

            neighbors.append((rr, cc))

    return neighbors


def get_neighbor(height_map: np.ndarray, rc: RC, max_step: float) -> Optional[RC]:
    neighbors = get_valid_neighbors(height_map, rc, max_step)
    if not neighbors:
        return None
    return random.choice(neighbors)


def step_distance(a: RC, b: RC, scale: float) -> float:
    ar, ac = a
    br, bc = b
    diagonal = (ar != br) and (ac != bc)
    return scale * (math.sqrt(2.0) if diagonal else 1.0)


def simulated_annealing(
    height_map: np.ndarray,
    start_rc: RC,
    *,
    n_iterations: int,
    max_step: float,
    temp: float,
    scale: float,
    seed: int = 0,
) -> Tuple[RC, float, List[RC], float]:
    """
    Returns:
      best_rc, best_height, path_taken (accepted moves), total_distance_m
    """
    random.seed(seed)

    sr, sc = start_rc
    if height_map[sr, sc] < 0:
        raise ValueError(f"Start {start_rc} is invalid (-1). Pick a different start.")

    best = start_rc
    best_eval = objective_function(height_map, best)

    current = best
    current_eval = best_eval

    path = [current]
    total_dist = 0.0

    for i in range(n_iterations):
        t = temp / float(i + 1)

        candidate = get_neighbor(height_map, current, max_step)
        if candidate is None:
            break

        candidate_eval = objective_function(height_map, candidate)

        if candidate_eval < best_eval or random.random() < math.exp((current_eval - candidate_eval) / t):
            total_dist += step_distance(current, candidate, scale)
            current, current_eval = candidate, candidate_eval
            path.append(current)

            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval

        if i % 10 == 0:
            print(f"Iteration {i}, Temperature {t:.4f}, Best Height {best_eval:.3f}")

    return best, best_eval, path, total_dist


def main() -> None:
    crater_map = np.load("crater_map.npy")
    nr, nc = crater_map.shape

    scale = 10.045020712681001
    max_step = 2.0

    required_xy = (3350.0, 5800.0)

    tests_xy = [
        required_xy,
        (3500.0, 5000.0),
        (3000.0, 4500.0),
        (1000.0, 8000.0),
        (5000.0, 3000.0),
        (3200.0, 5200.0),
    ]

    print("=== Simulated Annealing ===")
    for xy in tests_xy:
        rc = xy_to_rc(*xy, nr=nr, scale=scale)
        r, c = rc

        start_h = float(crater_map[r, c])

        sa_best_rc, sa_best_h, sa_path, sa_dist = simulated_annealing(
            crater_map,
            rc,
            n_iterations=100,
            max_step=max_step,
            temp=10.0,
            scale=scale,
            seed=0,
        )

        sa_best_xy = rc_to_xy(sa_best_rc[0], sa_best_rc[1], nr=nr, scale=scale)
        steps_taken = max(0, len(sa_path) - 1)
        height_drop = sa_best_h - start_h

        print(
            f"start_xy={xy} start_h={start_h:.3f}\n"
            f"best_h={sa_best_h:.3f} change_in_height={height_drop:.3f}\n"
            f"steps={steps_taken} dist_m={sa_dist:.1f} best_xy={sa_best_xy}\n"
        )


if __name__ == "__main__":
    main()