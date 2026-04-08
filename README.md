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

# CloudControl-Env (CORE-RL)

An OpenEnv Benchmark for Agentic Cloud Cost Optimization (FinOps).

## Real-World Task Simulation
This environment simulates a cloud billing and resource dashboard. AI agents act as FinOps Engineers to find "zombie" resources, right-size infrastructure, and manage budgets without stopping critical production nodes.

## Setup & Usage
1. Build and run the environment: `docker build -t core-rl . && docker run -p 7860:7860 core-rl`
2. Run the baseline: `python inference.py`

## Action Space
- `stop`: Terminates a resource.
- `resize`: Downgrades a resource to a cheaper tier.
- `no_op`: Takes no action.

## Observation Space
- `resources`: List of active cloud objects with cost/utilization metrics.
- `current_hourly_spend`: Total burn rate.
- `budget_limit`: The target spending cap.