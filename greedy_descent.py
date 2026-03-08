from __future__ import annotations

from datetime import datetime

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

RC = Tuple[int, int]


@dataclass
class GreedyResult:
    start_rc: RC
    end_rc: RC
    path: List[RC]
    distance_m: float
    start_height: float
    end_height: float
    steps: int
    stopped_reason: str


def xy_to_rc(x_m: float, y_m: float, *, nr: int, scale: float) -> RC:
    c = int(round(x_m / scale))
    r = int(nr - round(y_m / scale))
    return r, c

def rc_to_xy(r: int, c: int, *, nr: int, scale: float) -> Tuple[float, float]:
    x_m = float(c) * float(scale)
    y_m = float(nr - r) * float(scale)
    return x_m, y_m

def rc_in_bounds(r: int, c: int, nr: int, nc: int) -> bool:
    return 0 <= r < nr and 0 <= c < nc


def greedy_descent(
    height: np.ndarray,
    start: RC,
    *,
    max_step_m: float = 2.0,
    scale_m_per_px: float = 10.045020712681001,
    max_iters: int = 2_000_000,
) -> GreedyResult:
    nr, nc = height.shape

    sr, sc = start

    path: List[RC] = [start]
    dist = 0.0

    cur = start
    cur_h = float(height[cur])

    # 8-neighborhood
    neigh = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]

    reason = "max_iters reached"
    for _ in range(max_iters):
        cr, cc = cur
        cur_h = float(height[cr, cc])

        best_next: Optional[RC] = None
        best_h = cur_h

        for dr, dc in neigh:
            rr, cc2 = cr + dr, cc + dc
            if not rc_in_bounds(rr, cc2, nr, nc):
                continue

            nh = float(height[rr, cc2])
            if nh < 0:
                continue

            if abs(nh - cur_h) > max_step_m:
                continue

            if nh < best_h:
                best_h = nh
                best_next = (rr, cc2)

        if best_next is None:
            reason = "no lower neighbor within step constraint"
            break

        # move
        nr2, nc2 = best_next
        step_cost = scale_m_per_px * (math.sqrt(2.0) if (nr2 != cr and nc2 != cc) else 1.0)
        dist += step_cost
        cur = best_next
        path.append(cur)

    return GreedyResult(
        start_rc=start,
        end_rc=cur,
        path=path,
        distance_m=dist,
        start_height=float(height[start]),
        end_height=float(height[cur]),
        steps=len(path) - 1,
        stopped_reason=reason,
    )



def main() -> None:
    print("run on ", datetime.now())
    # Load the matrix
    crater_map = np.load("crater_map.npy")
    nr, nc = crater_map.shape
    print(nr, "is nr, the following is nc ", nc)

    scale = 10.045020712681001  # meters/pixel
    max_step = 2.0   # meters

    # Required test position
    start_xy = (3350.0, 5800.0)
    start_rc = xy_to_rc(*start_xy, nr=nr, scale=scale)

    res = greedy_descent(crater_map, start_rc, max_step_m=max_step, scale_m_per_px=scale)

    end_xy = rc_to_xy(res.end_rc[0], res.end_rc[1], nr=nr, scale=scale)

    print("=== Given coordinates ===")
    print(f"start_xy={start_xy} -> start_rc={res.start_rc}, start_height={res.start_height:.3f}")
    print(f"end_xy={end_xy} -> end_rc={res.end_rc}, end_height={res.end_height:.3f}")
    print(f"change_in_height={res.end_height - res.start_height}, steps={res.steps}, distance_m={res.distance_m:.2f}")
    print(f"stopped_reason={res.stopped_reason}")


    tests_xy = [
        (3500.0, 5000.0),
        (3000.0, 4500.0),
        (1000.0, 8000.0),
        (5000.0, 3000.0),
        (3200.0, 5200.0),
    ]

    print("\n=== Additional coordinates ===")
    for xy in tests_xy:
        rc = xy_to_rc(*xy, nr=nr, scale=scale)
        r, c = rc
        start_height = float(crater_map[r, c])
        run = greedy_descent(crater_map, rc, max_step_m=max_step, scale_m_per_px=scale)
        print(f"start_xy={xy} -> start_rc={rc}, start_height={start_height:.3f}, ")
        print(f"end_height={run.end_height:.3f}, change_in_height={run.end_height - start_height}, dist_m={run.distance_m:.1f}, steps={run.steps}")
        print()


if __name__ == "__main__":
    main()