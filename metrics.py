# Creating the new `metrics.py` module for SCOPUS-standard evaluation metrics

import numpy as np
import math
import csv
from typing import List, Tuple, Optional

def compute_total_distance(path: List[Tuple[int, int]]) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))

def compute_inflection_points(path: List[Tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    count = 0
    for i in range(1, len(path)-1):
        dx1 = path[i][0] - path[i-1][0]
        dy1 = path[i][1] - path[i-1][1]
        dx2 = path[i+1][0] - path[i][0]
        dy2 = path[i+1][1] - path[i][1]
        if (dx1, dy1) != (dx2, dy2):
            count += 1
    return count

def compute_total_turning_angle(path: List[Tuple[int, int]]) -> float:
    if len(path) < 3:
        return 0.0
    angle_sum = 0.0
    for i in range(1, len(path)-1):
        v1 = np.array([path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]])
        v2 = np.array([path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
        cosine_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = math.degrees(math.acos(cosine_angle))
        angle_sum += angle
    return angle_sum

def evaluate_path(path: Optional[List[Tuple[int, int]]], runtime_s: float, nodes_explored: int) -> dict:
    if not path:
        return {
            "path_length": 0,
            "total_distance": 0.0,
            "inflection_points": 0,
            "total_turning_angle": 0.0,
            "runtime_s": runtime_s,
            "nodes_explored": nodes_explored
        }
    return {
        "path_length": len(path),
        "total_distance": compute_total_distance(path),
        "inflection_points": compute_inflection_points(path),
        "total_turning_angle": compute_total_turning_angle(path),
        "runtime_s": runtime_s,
        "nodes_explored": nodes_explored
    }

def export_metrics_to_csv(metrics_list: List[dict], filename: str):
    if not metrics_list:
        return
    keys = metrics_list[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_list)
