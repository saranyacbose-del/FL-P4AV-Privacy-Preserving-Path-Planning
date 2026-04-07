# FL-P4AV: Privacy-Preserving Federated Learning for Autonomous Robot Path Planning

## Description

FL-P4AV (Federated Learning-based Privacy-Preserving Personalized Path Planning for Autonomous Vehicles) is a decentralized framework that enables collaborative autonomous robotic systems to compute personalized navigation paths without sharing raw sensor data.

The framework integrates four core components:

- **Lightweight semantic cost prediction model** — each vehicle trains a linear model on local environmental features (obstacle density, goal proximity, normalized spatial coordinates) without transmitting raw data.
- **Federated aggregation (FedAvg)** — a central server collects privacy-protected model parameters and constructs a shared global model through weighted averaging.
- **Differential privacy module** — calibrated Gaussian noise is applied to model parameters before transmission, providing formal (ε, δ)-differential privacy guarantees.
- **Semantic-Aware A\* planner** — integrates predicted traversal costs into the A\* heuristic function, enabling context-aware and personalized route planning. A dynamic obstacle handling module triggers real-time path recalculation without centralized control.

Experiments across grid environments of three scales (20×20, 30×30, 50×50), with spatial configurations inspired by the Waymo Open Dataset and the KITTI Stereo Dataset, show that FL-P4AV achieves an average path length of 20.8 steps and planning latency below 0.003 seconds, outperforming five baseline methods (Static A\*, Q-Learning+MPC, FLPTM, Behrens et al., and FL-NoDP) across all evaluation metrics.

---

## Dataset Information

FL-P4AV does **not** redistribute or require downloading any external dataset. The framework uses **simulated grid environments** whose structural configurations and obstacle distributions are *inspired by* two publicly available datasets:

### 1. KITTI Stereo Dataset
- **Source:** Geiger A, Lenz P, Stiller C, Urtasun R (2013). Vision meets robotics: The KITTI dataset. *International Journal of Robotics Research*, 32(11):1231–1237.
- **DOI:** https://doi.org/10.1177/0278364913491297
- **URL:** http://www.cvlibs.net/datasets/kitti/
- **How used:** Stereo depth cues and obstacle layout patterns from KITTI sequences were used to inform obstacle density distributions and grid structure in the 30×30 and 50×50 simulation environments. Raw KITTI images and point clouds were **not** used directly; only spatial statistics (obstacle frequency, spatial clustering patterns) were abstracted into grid-based representations.

### 2. Waymo Open Dataset
- **Source:** Waymo LLC (2025). Waymo Open Dataset. Available at https://waymo.com/open/ (Accessed March 2025).
- **URL:** https://waymo.com/open/
- **How used:** Waymo scene-level annotations (road occupancy maps, pedestrian and vehicle positions) informed the obstacle placement logic in the 20×20 simulation grids, representing urban driving scenarios. Raw Waymo sensor data was **not** used directly; grid abstractions were constructed programmatically.

### Data Preprocessing
The preprocessing pipeline converts dataset-inspired spatial statistics into discrete grid environments as follows:

1. Obstacle density ratios were estimated from representative KITTI and Waymo scenes.
2. Grid cells were randomly populated with obstacles according to these density ratios, bounded to ensure path feasibility: obstacle count O < (N² − 2) for an N×N grid.
3. Source and destination cells were assigned to obstacle-free positions.
4. Each vehicle's local feature vector per cell consists of: obstacle density (local 3×3 window), normalized Euclidean distance to goal, and normalized (row, column) coordinates — four scalar features in total.
5. No normalization beyond coordinate scaling [0, 1] was applied to input features.

No raw data files from either dataset are included in this repository.

---

## Code Information

```
FL-P4AV/
│
├── main.py                              # Entry point: runs full FL-P4AV simulation
├── federated_path_planning.py           # Core FL training loop (initial version)
├── federated_path_planning_updated.py   # Revised FL loop with DP noise
├── federated_path_planning_updated2.py  # Final version with semantic A* integration
├── metrics.py                           # Evaluation metrics computation
├── requirements.txt                     # Python dependencies
│
├── models/                              # Semantic cost model definition
├── federated/                           # FedAvg aggregation logic
├── privacy/                             # Gaussian noise injection (DP module)
├── planner/                             # Semantic-Aware A* implementation
├── experiments/                         # Simulation configurations and run scripts
└── results/                             # Output plots and result tables
```

### File Descriptions

| File | Description |
|---|---|
| `main.py` | Orchestrates the full pipeline: environment setup, federated training, path planning, and evaluation |
| `federated_path_planning_updated2.py` | Primary implementation — use this for reproducing paper results |
| `metrics.py` | Computes path length, total distance, inflection points, turning angle, planning time, nodes explored, and privacy budget ε |
| `models/` | Linear semantic cost model: predicts traversal cost from 4 local features per grid cell |
| `federated/` | FedAvg aggregation: computes weighted average of parameters from N vehicles |
| `privacy/` | Gaussian mechanism: adds noise calibrated to sensitivity S and σ before parameter upload |
| `planner/` | Modified A\* that uses predicted semantic cost as an additive term in f(n) = g(n) + h(n) + λ·cost(n) |
| `experiments/` | Simulation scripts for 20×20, 30×30, and 50×50 grids; scalability experiments (5–100 vehicles) |

---

## Requirements

Python 3.8 or higher is required.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
```

> **Note:** No deep learning framework (PyTorch, TensorFlow) is required. The semantic cost model is a lightweight linear model implemented entirely in NumPy.

---

## Usage Instructions

### 1. Clone the repository

```bash
git clone https://github.com/saranyacbose-del/FL-P4AV-Privacy-Preserving-Path-Planning.git
cd FL-P4AV-Privacy-Preserving-Path-Planning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full simulation (reproduces paper results)

```bash
python main.py
```

This runs FL-P4AV across all three grid sizes (20×20, 30×30, 50×50) with default parameters and outputs evaluation metrics to the `results/` directory.

### 4. Run individual experiments

```bash
python experiments/run_simulation.py
```

### 5. Key configuration parameters

The following parameters can be adjusted in the simulation scripts:

| Parameter | Default | Description |
|---|---|---|
| Grid size (N) | 20, 30, 50 | Size of N×N navigation grid |
| Number of vehicles | 10 | Federated clients |
| Federated rounds | 10 | Number of FL communication rounds |
| Local epochs | 5 | Local training iterations per round |
| DP noise std (σ) | 0.1 | Gaussian noise standard deviation |
| Privacy δ | 1e-5 | DP failure probability |
| Semantic weight (λ) | 1.0 | Weight of semantic cost in A\* evaluation |

### 6. Outputs

All results are saved in the `results/` directory:
- `path_metrics.csv` — per-trial path length, distance, inflection points, turning angle, planning time
- `convergence_plot.png` — federated model loss across rounds
- `privacy_utility_tradeoff.png` — path quality vs. privacy budget ε
- `scalability.png` — communication overhead vs. fleet size

---

## Methodology

The FL-P4AV pipeline proceeds as follows:

**Step 1 — Environment Initialization**
An N×N grid G is constructed with semantic labels: 0 (free), 1 (obstacle), 2 (source), 3 (destination). Obstacle positions are sampled uniformly from valid cells with count bounded to ensure feasibility.

**Step 2 — Local Feature Extraction**
Each vehicle i extracts a 4-dimensional feature vector per cell: [obstacle_density, goal_distance, norm_row, norm_col].

**Step 3 — Local Model Training**
Each vehicle trains a linear regression model w_i (weights over 4 features) using its local grid observations as training samples, minimizing mean squared error between predicted and target traversal costs.

**Step 4 — Differential Privacy Noise Injection**
Before transmission, Gaussian noise η ~ N(0, σ²·I) is added to model parameters w_i. The noise standard deviation σ is selected to satisfy (ε, δ)-differential privacy using the Gaussian mechanism, where ε ≈ 0.043 for σ = 0.1 and δ = 1e-5.

**Step 5 — Federated Aggregation (FedAvg)**
The server computes the global model as a weighted average: w_global = Σ (n_i / N_total) · w̃_i, where w̃_i is the noisy parameter vector from vehicle i.

**Step 6 — Local Model Update (Weighted Interpolation)**
Each vehicle updates its local model: w_i ← α · w_global + (1 − α) · w_i, where α is the interpolation weight controlling the degree of global adaptation.

**Step 7 — Semantic Cost Map Generation**
Each vehicle applies its updated model to predict traversal costs for all free cells in the grid, producing a semantic cost map C ∈ R^(N×N).

**Step 8 — Semantic-Aware A\* Path Planning**
The modified A\* evaluates each node n as: f(n) = g(n) + h(n) + λ · C(n), where g(n) is path cost so far, h(n) is the Euclidean heuristic to the goal, and C(n) is the predicted semantic traversal cost.

**Step 9 — Dynamic Obstacle Re-planning**
If a dynamic obstacle appears on the current planned path, the A\* search is re-triggered from the vehicle's current position with the updated obstacle map — without any centralized coordination.

**Step 10 — Evaluation**
Path quality is measured across all vehicles and grid configurations using the metrics described below.

---

## Evaluation Methodology

### Comparative Analysis
FL-P4AV is compared against five baselines:

| Baseline | Description |
|---|---|
| Static A\* | Standard A\* with no semantic cost and no federated learning |
| Q-Learning + MPC | Hierarchical RL-based planner (Gong et al., 2024) |
| FLPTM | Federated learning for traffic management adapted for path planning (Alsharif et al., 2024) |
| Behrens et al. | Privacy-preserving FL for feedforward control in multi-agent systems |
| FL-NoDP | FL-P4AV without differential privacy (ablation) |

### Ablation Study
To isolate the contribution of each component, the following ablated configurations are evaluated:

- **FL-NoDP:** Removes the differential privacy module (σ = 0)
- **No-Semantic:** Replaces semantic cost with uniform cost (λ = 0), reducing to standard federated A\*
- **No-Federation:** Each vehicle uses only its local model (no FedAvg aggregation)
- **No-Replanning:** Disables dynamic obstacle re-planning

### Assessment Metrics

| Metric | Unit | Justification |
|---|---|---|
| Path Length | Steps | Primary measure of navigation efficiency; shorter paths reduce energy use and travel time |
| Total Distance | Grid units | Accounts for diagonal moves; complements path length |
| Inflection Points | Count | Measures path smoothness; fewer turns improve vehicle stability |
| Turning Angle | Degrees | Quantifies sharpness of direction changes; relevant for kinematic constraints |
| Planning Time | Seconds | Measures real-time viability; must remain below control loop frequency |
| Nodes Explored | Count | Measures search efficiency of the A\* planner |
| Privacy Budget (ε) | — | Formal DP guarantee; lower ε = stronger privacy |
| Communication Overhead | Bytes | Ratio of parameter size to raw sensor data size; measures scalability |

Statistical significance of all pairwise comparisons is confirmed via paired Wilcoxon signed-rank tests (p < 0.05).

---

## Citations

If you use this code or dataset in your research, please cite:

```
Saranya C, Janaki G (2026). Privacy-Preserving Federated Learning for Autonomous 
Robot Path Planning. PeerJ Computer Science [under review].
GitHub: https://github.com/saranyacbose-del/FL-P4AV-Privacy-Preserving-Path-Planning
```

### Dataset references

```
Geiger A, Lenz P, Stiller C, Urtasun R (2013). Vision meets robotics: The KITTI dataset.
International Journal of Robotics Research, 32(11):1231–1237.
https://doi.org/10.1177/0278364913491297

Waymo LLC (2025). Waymo Open Dataset. https://waymo.com/open/ (Accessed March 2025).
```

### Key algorithmic references

```
McMahan B, Moore E, Ramage D, Hampson S, Arcas BAy (2017). Communication-efficient 
learning of deep networks from decentralized data. AISTATS 2017, pp 1273–1282.

Dwork C, Roth A (2014). The algorithmic foundations of differential privacy. 
Foundations and Trends in Theoretical Computer Science, 9(3–4):211–407.
```

---

## License & Contribution Guidelines

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for full terms.

### Contributions

Contributions, bug reports, and feature suggestions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

For questions or collaborations, contact: saranyaresearch22@gmail.com

---

## Authors

- **Saranya C** — Conceptualization, methodology, algorithm design, software implementation, data curation, formal analysis, visualization, writing
- **Janaki G** — Supervision, validation, investigation, resources, review and editing

Department of Electrical and Electronics Engineering,
SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India.
