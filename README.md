# Agentic Workflow RL System

A complete reinforcement learning framework combining **Deep Q-Networks (DQN)**, **Transfer Learning**, **Retrieval-Augmented Generation (RAG)**, and **Multi-Agent Communication**. Built as part of the postgraduate dissertation work for MSc in Artificial Intelligence and Data Analytics at Loughborough University London.

Author: **Divya Teja Mannava**

---

## 🧠 Project Overview

This project explores how reinforcement learning agents can be enhanced using memory, retrieval, and inter-agent communication mechanisms. The core architecture supports:

* ✅ Deep Q-Networks (DQN) for RL training
* ♻️ Transfer Learning: reuse learned policies
* 🧠 RAG + RL hybrid: retrieval-based context conditioning
* 🔄 Multi-Agent Communication with messaging
* 📈 Logging and visualisation of performance metrics
* 🔬 Extensible with PPO, UCB, and other advanced methods

This work is designed to be modular, scalable, and publication-ready.

---

## 📂 Folder Structure

```
Agentic_Workflow_RL/
├── agents/                  # DQN and other RL agents
│   ├── dqn.py               
│   └── transfer_dqn.py      # Transfer Learning wrapper
├── envs/                    # Custom environment logic
├── memory/                  # Replay buffer, RAG store
├── policies/                # Policy selection strategies
├── trainer.py               # Main training loop
├── utils/
│   ├── log_utils.py         # Logging to CSV
│   └── plot_metrics.py      # Plots: reward/loss
├── logs/
│   └── metrics.csv          # Training data
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 🚀 Setup Instructions

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

Note: If `gym` is deprecated, switch to Gymnasium:

```bash
pip install gymnasium
```

---

### 2. Run Training (DQN)

```bash
python trainer.py
```

The agent will train over 200 episodes and save logs in `logs/metrics.csv`.

---

### 3. Plot Reward & Loss Curves

```bash
python utils/plot_metrics.py
```

This will generate `training_metrics_plot.png` comparing reward and loss across episodes.

---

## 📊 Evaluation & Logging

* **Reward**: Cumulative score per episode
* **Loss**: Average MSE over Q-value updates
* **Epsilon**: Exploration rate (decays over time)
* **CSV Logger**: Auto-logs to `metrics.csv`
* **Plot Script**: Generates clean performance graphs

---

## 🔁 Transfer Learning Support

A pre-trained DQN can be loaded and reused on new environments. The `transfer_dqn.py` module wraps policy import, selective layer freezing, and reinitialisation logic.

---

## 🧠 Agent Communication & Memory (Advanced)

Future iterations include:

* LSTM-based memory and message passing
* RAG-enabled policy retrieval
* Agent coordination for collaborative learning

---

## 🧪 Experimental Design

* Environment: `CartPole-v1`
* Episodes: 200
* Replay Buffer: 10,000 steps
* Batch Size: 64
* Target Network Sync: Every episode
* Optimizer: Adam
* Loss: MSELoss

---

## 📌 Author

**Divya Teja Mannava

Dissertation: *"Agentic Multi-Agent RL Systems with Memory, Communication, and Retrieval-Augmented Reasoning"*

---

## 📃 License

MIT License (or your preferred open-source license)
