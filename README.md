# 🚗 FL-P4AV: Privacy-Preserving Federated Learning for Personalized Path Planning

## 📌 Overview

FL-P4AV is a **Federated Learning-based framework with Differential Privacy** designed for **personalized path planning in collaborative autonomous ground vehicles (AGVs)**.

The system enables multiple vehicles to collaboratively learn optimal navigation strategies **without sharing raw data**, ensuring **privacy preservation, scalability, and adaptability** in dynamic environments.

---

## 🎯 Key Features

* 🔁 **Federated Learning (FedAvg)** for collaborative model training
* 🔐 **Differential Privacy (Gaussian Mechanism)** for secure parameter sharing
* 🧠 **Semantic Cost Modeling** using lightweight local models
* 🛣️ **Semantic-Aware A*** path planning
* 🔄 **Dynamic Obstacle Handling & Real-time Re-planning**
* 📊 Comprehensive evaluation metrics and visualizations

---

## 🏗️ Project Structure

```
FL-P4AV/
│
├── data/                # Sample grid data / environment generation
├── models/              # Local semantic cost model
├── federated/           # Federated learning (aggregation & training)
├── privacy/             # Differential privacy (noise injection)
├── planner/             # Semantic-aware A* implementation
├── experiments/         # Simulation scripts and configs
├── results/             # Output plots and tables
│
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/FL-P4AV-Privacy-Preserving-Path-Planning.git
cd FL-P4AV-Privacy-Preserving-Path-Planning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the full simulation:

```bash
python main.py
```

Run experiments separately:

```bash
python experiments/run_simulation.py
```

---

## 🧪 Experimental Configuration

| Parameter           | Value               |
| ------------------- | ------------------- |
| Grid Sizes          | 20×20, 30×30, 50×50 |
| Vehicles            | 5–100               |
| Federated Rounds    | 10                  |
| Local Epochs        | 5                   |
| Noise Std (σ)       | 0.1                 |
| Privacy δ           | 1e-5                |
| Semantic Weight (λ) | 1.0                 |

---

## 📊 Output Metrics

The framework evaluates:

* Path Length
* Total Distance
* Inflection Points
* Turning Angle
* Planning Time
* Nodes Explored
* Privacy Budget (ε)

All outputs are stored in the `results/` directory.

---

## 🔐 Differential Privacy Details

* Mechanism: Gaussian Noise Injection
* Sensitivity: Computed from model updates
* Privacy Budget:

  * ε ≈ 0.043 (σ = 0.1, δ = 1e-5)
* Ensures protection against:

  * Gradient inference attacks
  * Membership inference

---

## 🧠 Methodology

1. Each vehicle trains a **local semantic cost model**
2. Model parameters are **perturbed using DP noise**
3. Parameters are sent to a **federated server**
4. Global model is computed via **FedAvg**
5. Vehicles update local models using **weighted interpolation**
6. Path planning is performed using **Semantic-Aware A***

---

## 🔄 Reproducibility

To reproduce results:

```bash
python experiments/run_simulation.py
```

* Random seeds are controlled
* Configurations stored in `configs.yaml`
* Results averaged over multiple trials

---

## 📈 Example Outputs

* Dynamic path planning visualizations
* Convergence plots
* Privacy-utility trade-off graphs

---

## 📂 Dataset

* Synthetic grid environments
* Inspired by:

  * Waymo Open Dataset
  * KITTI Stereo Dataset

(No raw dataset is redistributed due to licensing)

---

## ⚠️ Limitations

* Uses **linear model** for cost prediction (lightweight but limited)
* Simulation-based evaluation (no real-world deployment yet)
* Assumes reliable communication between agents

---

## 🚀 Future Work

* Deep learning-based semantic models
* Real-world autonomous vehicle deployment
* Communication-efficient federated learning
* Robustness against adversarial attacks

---

## 📜 Citation

If you use this work, please cite:

```
Saranya C, Janaki G,
"FL-P4AV: A Federated Learning Framework With Differential Privacy for Personalized Path Planning in Collaborative Autonomous Ground Vehicles",
```

---

## 👩‍💻 Authors

* Saranya C
* Janaki G

---

## 📬 Contact

For queries or collaborations:
📧 [saranyaresearch22@gmail.com](mailto:saranyaresearch22@gmail.com)

---

## ⭐ Acknowledgment

This work was developed as part of research at
**SRM Institute of Science and Technology, India**

---

## 📢 License

This project is released for academic and research purposes.
