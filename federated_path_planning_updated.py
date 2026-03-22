#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Learning-Enabled Privacy-Preserving Personalized Path Planning
for Collaborative Autonomous Ground Vehicles

SCIE-ready version 1.3  ·  13 May 2025
Author : Dr John Blesswin A  <johnb@srmist.edu.in>
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ╔═══════════════════════════════════════════════════════════════╗
# ║ 1 ▸ CONFIGURATION                                             ║
# ╚═══════════════════════════════════════════════════════════════╝
@dataclass
class SimConfig:
    grid_size: int = 20
    num_obstacles: int = 50
    num_vehicles: int = 5
    federated_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 1e-2
    dp_noise: float = 0.1
    dynamic_obstacle_prob: float = 0.05
    timesteps: int = 5
    seed: int = 42
    output_dir: str = "results"
    visualize: bool = True
    dpi: int = 300
    legend_fontsize: int = 9


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 2 ▸ ENVIRONMENT                                               ║
# ╚═══════════════════════════════════════════════════════════════╝
class Environment:
    """Grid world with static & dynamic obstacles."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.size = cfg.grid_size
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self._init_grid()

    # ------------------------------------------------------------------
    def _init_grid(self) -> None:
        rng = random.Random(self.cfg.seed)
        self.grid[:] = 0

        # global reference start/goal (tinted, but each vehicle gets its own icons)
        sx, sy = rng.randint(0, self.size // 4), rng.randint(0, self.size // 4)
        dx, dy = rng.randint(3 * self.size // 4, self.size - 1), rng.randint(
            3 * self.size // 4, self.size - 1
        )
        self.source = (sx, sy)
        self.destination = (dx, dy)
        self.grid[sx, sy] = 2
        self.grid[dx, dy] = 3

        placed = 0
        while placed < self.cfg.num_obstacles:
            x, y = rng.randint(0, self.size - 1), rng.randint(0, self.size - 1)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 1
                placed += 1

    # ------------------------------------------------------------------
    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] != 1

    # ------------------------------------------------------------------
    def neighbours(self, x: int, y: int):
        moore = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1))
        return [(x+dx, y+dy) for dx,dy in moore if self.is_valid(x+dx, y+dy)]

    # ------------------------------------------------------------------
    def move_dynamic_obstacles(self) -> None:
        rng = random.Random()
        xs, ys = np.where(self.grid == 1)
        for x, y in zip(xs, ys):
            if rng.random() < self.cfg.dynamic_obstacle_prob:
                self.grid[x, y] = 0
                for _ in range(10):
                    nx = max(0, min(self.size - 1, x + rng.randint(-1, 1)))
                    ny = max(0, min(self.size - 1, y + rng.randint(-1, 1)))
                    if self.grid[nx, ny] == 0:
                        self.grid[nx, ny] = 1
                        break
                else:
                    self.grid[x, y] = 1

    # ------------------------------------------------------------------
    def save_csv(self, name: str) -> None:
        p = os.path.join(self.cfg.output_dir, name)
        np.savetxt(p, self.grid, fmt="%d", delimiter=",")
        logger.info("Grid saved → %s", p)

    # ------------------------------------------------------------------
    def plot(
        self,
        paths: Optional[List[Sequence[Tuple[int, int]]]] = None,
        title: str = "Figure",
        figure_id: int = 1,
    ) -> None:
        """SCIE-quality plot with per-vehicle start & goal icons."""
        cfg = self.cfg
        colour_grid = np.zeros((*self.grid.shape, 3))
        colour_grid[self.grid == 0] = [1, 1, 1]
        colour_grid[self.grid == 1] = [0, 0, 0]
        colour_grid[self.grid == 2] = [1, 0.85, 0.85]
        colour_grid[self.grid == 3] = [0.85, 1, 0.85]

        plt.figure(figsize=(6, 6), dpi=cfg.dpi)
        plt.imshow(colour_grid, origin="upper", zorder=0)
        plt.grid(True, color="lightgrey", linewidth=0.4, zorder=1)
        plt.xlabel("Y-axis (columns)")
        plt.ylabel("X-axis (rows)")
        plt.title(title, pad=12)

        # colour & linestyle cycles
        colours = ["royalblue", "firebrick", "darkorange", "purple", "deepskyblue",
                   "gold", "lime", "hotpink", "sienna", "teal"]
        linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

        # plot each vehicle’s path + markers
        handles = []
        if paths:
            for idx, p in enumerate(paths):
                if not p:
                    continue
                xs, ys = zip(*p)
                col = colours[idx % len(colours)]
                style = linestyles[idx % len(linestyles)]
                plt.plot(
                    ys,
                    xs,
                    linestyle=style,
                    color=col,
                    lw=1.8,
                    zorder=3,
                )
                # start marker
                sx, sy = p[0]
                plt.scatter(
                    [sy],
                    [sx],
                    marker="o",
                    c=col,
                    s=90,
                    edgecolors="k",
                    linewidths=0.4,
                    zorder=4,
                )
                # destination marker
                dx, dy = p[-1]
                plt.scatter(
                    [dy],
                    [dx],
                    marker="D",
                    c=col,
                    s=90,
                    edgecolors="k",
                    linewidths=0.4,
                    zorder=4,
                )
                handles.append(
                    plt.Line2D(
                        [], [],
                        color=col,
                        lw=2,
                        linestyle=style,
                        markerfacecolor=col,
                        markeredgecolor="k",
                        marker="o",
                        markersize=6,
                        label=f"V{idx} (S,G,path)",
                    )
                )

        plt.legend(
            handles=handles,
            fontsize=cfg.legend_fontsize,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
        )

        fname = f"SCIE_Figure_{figure_id:02d}_{title.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(
            os.path.join(cfg.output_dir, fname),
            dpi=cfg.dpi,
            bbox_inches="tight",
        )
        if cfg.visualize:
            plt.show()
        else:
            plt.close()
        logger.info("Saved %s", fname)


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 3 ▸ PATHFINDER                                                ║
# ╚═══════════════════════════════════════════════════════════════╝
class AStar:
    def __init__(self, env: Environment) -> None:
        self.env = env

    @staticmethod
    def _h(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def find(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        wmap: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        import heapq

        open_set: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g: Dict[Tuple[int, int], float] = {start: 0.0}
        f: Dict[Tuple[int, int], float] = {start: self._h(start, goal)}
        explored = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            explored += 1
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, explored

            for n in self.env.neighbours(*current):
                tentative = g[current] + 1 + (wmap.get(n, 0.0) if wmap else 0.0)
                if tentative < g.get(n, float("inf")):
                    came_from[n] = current
                    g[n] = tentative
                    f[n] = tentative + self._h(n, goal)
                    heapq.heappush(open_set, (f[n], n))
        return None, explored


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 4 ▸ LIGHTWEIGHT LOCAL MODEL                                   ║
# ╚═══════════════════════════════════════════════════════════════╝
class LightweightModel:
    def __init__(self, input_dim: int, rng: random.Random) -> None:
        self.w = rng.normalvariate(0, 0.1) * np.ones(input_dim)
        self.b = rng.normalvariate(0, 0.1)
        self.rng = rng

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(x, self.w) + self.b)

    def train(self, xs: np.ndarray, ys: np.ndarray, lr: float) -> None:
        bs = min(32, len(xs))
        idx = np.random.permutation(len(xs))
        xs, ys = xs[idx], ys[idx]
        for i in range(0, len(xs), bs):
            xb, yb = xs[i:i+bs], ys[i:i+bs]
            err = xb @ self.w + self.b - yb
            self.w -= lr * (err[:, None] * xb).mean(axis=0)
            self.b -= lr * err.mean()

    def noisy_weights(self, sigma: float) -> Tuple[np.ndarray, float]:
        return (
            self.w + self.rng.normalvariate(0, sigma) * np.ones_like(self.w),
            self.b + self.rng.normalvariate(0, sigma),
        )

    def load_global(self, wg: np.ndarray, bg: float, alpha: float = 0.8) -> None:
        self.w = alpha * self.w + (1 - alpha) * wg
        self.b = alpha * self.b + (1 - alpha) * bg


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 5 ▸ VEHICLE                                                   ║
# ╚═══════════════════════════════════════════════════════════════╝
class Vehicle:
    def __init__(self, vid: int, env: Environment, cfg: SimConfig) -> None:
        self.id = vid
        self.env = env
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + vid)

        sx, sy = env.source
        self.pos = (
            max(0, min(env.size-1, sx + self.rng.integers(-2, 3))),
            max(0, min(env.size-1, sy + self.rng.integers(-2, 3))),
        )

        self.planner = AStar(env)
        self.model = LightweightModel(4, random.Random(cfg.seed + vid))
        self.local_data = self._generate_local_data()

    # ------------------------------------------------------------------
    def _obstacle_density(self, x, y, r=2):
        sub = self.env.grid[max(0,x-r):x+r+1, max(0,y-r):y+r+1]
        return float((sub == 1).sum()) / sub.size

    # ------------------------------------------------------------------
    def _generate_local_data(self, n=120):
        ow, lw = self.rng.uniform(0.5,2), self.rng.uniform(0.5,1.5)
        xs, ys = [], []
        for _ in range(n):
            x, y = self.rng.integers(0, self.env.size, size=2)
            od = self._obstacle_density(x, y)
            dist = math.hypot(x-self.env.destination[0], y-self.env.destination[1])
            feat = np.array([x/self.env.size, y/self.env.size, od, dist/self.env.size])
            xs.append(feat)
            ys.append(ow*od + lw*(dist/self.env.size))
        return {"x": np.array(xs), "y": np.array(ys)}

    # ------------------------------------------------------------------
    def train_local(self):
        self.model.train(self.local_data["x"], self.local_data["y"], self.cfg.learning_rate)

    # ------------------------------------------------------------------
    def _weight_map(self):
        wm = {}
        for x in range(self.env.size):
            for y in range(self.env.size):
                if self.env.grid[x,y] in (1,2,3):
                    continue
                od = self._obstacle_density(x, y)
                dist = math.hypot(x-self.env.destination[0], y-self.env.destination[1])
                feat = np.array([x/self.env.size, y/self.env.size, od, dist/self.env.size])
                wm[(x,y)] = max(0.0, self.model.predict(feat))
        return wm

    # ------------------------------------------------------------------
    def plan(self):
        t0 = time.perf_counter()
        path, explored = self.planner.find(self.pos, self.env.destination, self._weight_map())
        t = time.perf_counter()-t0
        return path, compute_metrics(path, explored, t)


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 6 ▸ METRIC UTILS                                              ║
# ╚═══════════════════════════════════════════════════════════════╝
def compute_metrics(path, explored, t):
    if not path:
        return defaultdict(float)
    plen = len(path)-1
    dist = sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(plen))
    infl, angle = 0, 0.0
    for i in range(1, plen):
        v1 = np.array(path[i]) - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        if not np.array_equal(v1, v2):
            infl += 1
            angle += abs(math.degrees(math.acos(
                np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
            )))
    return {
        "path_length": float(plen),
        "total_distance": float(dist),
        "inflection_points": float(infl),
        "total_turning_angle": float(angle),
        "nodes_explored": float(explored),
        "planning_time_s": t,
        "runtime_s": t,
    }


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 7 ▸ FEDERATED COORDINATOR                                     ║
# ╚═══════════════════════════════════════════════════════════════╝
class FedCoordinator:
    def __init__(self, vehicles: List[Vehicle], cfg: SimConfig) -> None:
        self.vehicles = vehicles
        self.cfg = cfg
        self.wg: Optional[np.ndarray] = None
        self.bg: Optional[float] = None

    # ------------------------------------------------------------------
    def run(self):
        logger.info("Federated learning (%d rounds)", self.cfg.federated_rounds)
        for r in range(1, self.cfg.federated_rounds+1):
            logger.info("Round %d", r)
            for v in self.vehicles:
                v.train_local()
            ws, bs = zip(*(v.model.noisy_weights(self.cfg.dp_noise) for v in self.vehicles))
            self.wg, self.bg = np.mean(ws, axis=0), float(np.mean(bs))
            for v in self.vehicles:
                v.model.load_global(self.wg, self.bg)
        logger.info("FL complete.")

    # ------------------------------------------------------------------
    def save_global(self):
        if self.wg is None:
            return
        fn = os.path.join(self.cfg.output_dir, "global_model.csv")
        with open(fn, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["param","value"])
            for i,v in enumerate(self.wg): w.writerow([f"w{i}",v])
            w.writerow(["bias", self.bg])
        logger.info("Saved global model → %s", fn)


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 8 ▸ SIMULATION                                                ║
# ╚═══════════════════════════════════════════════════════════════╝
class Simulation:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.env = Environment(cfg)
        self.vehicles = [Vehicle(i, self.env, cfg) for i in range(cfg.num_vehicles)]
        self.coordinator = FedCoordinator(self.vehicles, cfg)
        self.metrics: List[Dict[str, float]] = []

    def _export_metrics(self):
        if not self.metrics:
            return
        fn = os.path.join(self.cfg.output_dir, "path_metrics.csv")
        with open(fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.metrics[0].keys())
            w.writeheader(); w.writerows(self.metrics)
        logger.info("Metrics → %s", fn)

    def run(self):
        logger.info("Simulation start")
        fid = 1
        self.env.plot(title="Initial Environment", figure_id=fid); fid+=1
        self.env.save_csv("environment_initial.csv")

        self.coordinator.run()

        paths = []
        for v in self.vehicles:
            p,m = v.plan()
            paths.append(p); m["vehicle"]=v.id; self.metrics.append(m)
            logger.info("Vehicle %s → len %s  time %.3f s", v.id, m["path_length"], m["planning_time_s"])
        self.env.plot(paths, "Planned Paths (t0)", fid); fid+=1
        self._export_metrics()

        for t in range(1, self.cfg.timesteps+1):
            self.env.move_dynamic_obstacles()
            dyn = []
            for v in self.vehicles:
                p,_ = v.plan(); dyn.append(p)
            self.env.plot(dyn, f"Dynamic Environment (t{t})", fid); fid+=1
            self.env.save_csv(f"environment_t{t}.csv")

        self.coordinator.save_global()
        logger.info("Simulation complete")


# ╔═══════════════════════════════════════════════════════════════╗
# ║ 9 ▸ CLI ENTRY-POINT                                           ║
# ╚═══════════════════════════════════════════════════════════════╝
def parse_args() -> SimConfig:
    a = argparse.ArgumentParser(description="Federated path-planning simulator (SCIE)")
    a.add_argument("--grid", type=int, default=20)
    a.add_argument("--vehicles", type=int, default=5)
    a.add_argument("--rounds", type=int, default=10)
    a.add_argument("--epochs", type=int, default=5)
    a.add_argument("--obstacles", type=int, default=50)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--timesteps", type=int, default=5)
    a.add_argument("--vis", action="store_true")
    a.add_argument("--out", default="results")
    p = a.parse_args()
    return SimConfig(
        grid_size=p.grid,
        num_vehicles=p.vehicles,
        federated_rounds=p.rounds,
        local_epochs=p.epochs,
        num_obstacles=p.obstacles,
        seed=p.seed,
        timesteps=p.timesteps,
        visualize=p.vis,
        output_dir=p.out,
    )


def main():
    cfg = parse_args()
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    Simulation(cfg).run()


if __name__ == "__main__":
    main()
