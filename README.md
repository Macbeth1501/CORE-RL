---
title: CloudControl-Env
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# 🚀 CORE-RL: Cloud Optimization & Resource Efficiency
### *A High-Fidelity OpenEnv Benchmark for Autonomous FinOps Agents*

**CORE-RL** is a realistic simulation of a modern cloud infrastructure dashboard. It challenges AI agents to act as **Autonomous FinOps Engineers**—navigating complex resource hierarchies to eliminate waste and enforce budgetary guardrails without ever compromising business-critical services.

---

## 💡 The "Why" (Real-World Utility)
Enterprise cloud waste is estimated at over **$30 Billion annually**. While static tools can flag waste, they cannot act. Human engineers are too slow to manage thousands of ephemeral resources. **CORE-RL** fills the "Agentic Gap" by providing a high-stakes environment where agents must balance mathematical optimization (saving money) with strict safety constraints (not deleting production databases).

---

## 🛠️ Environment Specifications

### **1. Observation Space (The Dashboard)**
The agent receives a state containing a fleet of `CloudResource` objects:
* **Resource ID:** Unique identifier (e.g., `srv-prod-01`).
* **Type:** `VM`, `Database`, or `Storage`.
* **Utilization:** CPU/Memory usage (0.0–1.0).
* **Cost:** Hourly spend in USD.
* **Criticality:** A boolean flag. If `True`, stopping this resource triggers an immediate terminal failure.

### **2. Action Space (The Auditor's Tools)**
* `stop`: Terminates the resource (Cost → $0.00).
* `resize`: Scales a resource down (Cost reduced by 50%).
* `no_op`: Maintains status quo.

---

## 🏆 The 3-Tier Task System
CORE-RL programmatically generates three distinct scenarios to test agent reasoning:

| Task | Level | Objective | Success Criteria |
| :--- | :--- | :--- | :--- |
| **zombie_hunter** | Easy | Find and terminate "Zombie" resources (0% utilization). | 100% of unused nodes stopped. |
| **fleet_resizer** | Medium | "Right-size" a cluster where nodes are oversized (<5% usage). | >30% savings while maintaining health. |
| **budget_breach** | Hard | Total spend is $1,200. Budget is $500. | Prune non-critical nodes to meet budget. |

---

## ⚖️ Multi-Component Reward Function
To ensure robust learning, the environment provides a dense reward signal:

sum R = R_{efficiency} + R_{bonus} - R_{penalty}

* **Efficiency (+0.2):** Awarded for every 10% reduction in total hourly spend.
* **Safety Penalty (-1.0):** Immediate Terminal Failure if a critical resource is stopped.
* **Stability Penalty (-0.1):** Applied per step to encourage decisive action.

---

## 🚀 Getting Started

### **1. Clone the Repository**
```bash
git clone [https://huggingface.co/spaces/Macbeth1501/core-rl](https://huggingface.co/spaces/Macbeth1501/core-rl)
cd core-rl
```

### **2. Install Dependencies**
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Set Environment Variables**
The baseline agent requires a Hugging Face token to access the LLM (Qwen-72B).
```bash
# Windows (Command Prompt)
set HF_TOKEN=your_huggingface_token_here

# Linux/Mac/PowerShell
export HF_TOKEN=your_huggingface_token_here
```

### **4. Run the Environment**
You can run the environment server using Docker or locally via Uvicorn.

**Option A: Docker (Recommended)**
```bash
docker build -t core-rl .
docker run -p 7860:7860 core-rl
```

**Option B: Local Python**
```bash
python -m uvicorn core_rl.server.app:app --host 0.0.0.0 --port 7860
```

### **5. Run the Baseline Agent**
In a separate terminal (ensure `HF_TOKEN` is set), run the inference script:
```bash
python inference.py
```

---

## 👥 Authors & Contributors
* **Rochan Awasthi** - [rochansawasthi@gmail.com](mailto:rochansawasthi@gmail.com)
* **Sayali Bambal** - [sayalibambal218@gmail.com](mailto:sayalibambal218@gmail.com)

---

## 🧠 Technical Architecture & Implementation

### **System Design**
CORE-RL follows the **OpenEnv standardized architecture**, decoupling the decision-making (Agent) from the world logic (Environment). 

* **Environment Layer (`server/core_rl_environment.py`):** A stateful simulation that manages a dynamic pool of cloud resources. It calculates real-time burn rates and enforces "Criticality" constraints.
* **API Layer (`server/app.py`):** A FastAPI-based wrapper that exposes the environment via a RESTful interface, enabling remote benchmarking.
* **Agentic Baseline (`inference.py`):** A high-reasoning agent utilizing **Qwen-2.5-72B-Instruct**. It uses a Few-Shot Prompting technique to parse complex JSON observations into atomic cloud actions.

### **Procedural Task Generation**
Unlike static benchmarks, CORE-RL uses **Procedural Generation**. Every time `/reset` is called, the environment generates a unique infrastructure topology. This ensures the agent cannot "memorize" resource IDs and must rely on actual numerical reasoning to identify waste.

### **Reward Shaping & Safety**
The environment utilizes **Reward Shaping** to guide the agent toward safe optimization:
1.  **Dense Rewards:** Small increments for every USD saved, providing immediate feedback.
2.  **Hard Constraints:** A catastrophic penalty ($-1.0$) for stopping `is_critical` resources, training the agent to prioritize system uptime over cost savings.
3.  **Efficiency Incentives:** A step penalty that discourages the agent from "wavering" and encourages reaching the goal in the fewest moves possible.

### **Why CORE-RL Wins?**
1. **Direct Utility:** Addresses a $30B corporate pain point.
2. **Safety-First:** Includes "Criticality" logic, the #1 concern for real-world AI deployment.
3. **Benchmark Grade:** Provides a reproducible scoring system for all 3 difficulty tiers