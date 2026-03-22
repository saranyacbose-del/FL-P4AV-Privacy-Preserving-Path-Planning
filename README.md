# FL-P4AV: Privacy-Preserving Federated Path Planning

## 📌 Overview

This repository implements the FL-P4AV framework for personalized path planning in autonomous ground vehicles using Federated Learning and Differential Privacy.

## ⚙️ Features

* Federated Learning (FedAvg)
* Differential Privacy (Gaussian noise)
* Semantic-Aware A* Path Planning
* Dynamic obstacle handling

## 🧪 Requirements

```bash
pip install -r requirements.txt
```

## ▶️ Run Experiment

```bash
python main.py
```

## 📊 Outputs

* Path length
* Planning time
* Privacy budget (ε)
* Visualization plots

## 🔐 Privacy Settings

* Noise std (σ): 0.1
* δ: 1e-5

## 📁 Dataset

Synthetic grid environments derived from Waymo/KITTI abstractions.

## 🔁 Reproducibility

All experiments can be reproduced using:

```bash
python experiments/run_simulation.py
```

## 📜 Citation

If you use this work, please cite:
(Your IEEE Access paper citation here)

## 👨‍💻 Authors

* Saranya C
* Janaki G
