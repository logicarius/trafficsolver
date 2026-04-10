---
title: TrafficSolver
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# TrafficSolver — Urban Traffic OpenEnv

An OpenEnv environment where an AI agent acts as a traffic engineer,
observing congested road networks and taking actions to reduce delay
and improve flow.

Built on top of UrbanFlow — a graph-based traffic simulation study.

## Tasks

| Task | Difficulty | Goal |
|---|---|---|
| `clear_bottleneck` | Easy | Reduce queue at main bottleneck by 50% |
| `rebalance_network` | Medium | Get LCI < 0.4 and CDR < 0.7 |
| `maintain_stability` | Hard | Keep total delay under 600 over 10 steps |

## Baseline Scores
| Task | Score |
|---|---|
| clear_bottleneck | 0.60 |
| rebalance_network | 0.85 |
| maintain_stability | 1.00 |
| **Average** | **0.82** |

## Actions the Agent Can Take
- `add_capacity` — increase road capacity
- `add_road` — add a new parallel road
- `reroute` — shift load from congested roads
- `adjust_signal` — change traffic signal timing
- `do_nothing` — observe without acting

## API Endpoints
- `POST /reset` — start new episode
- `POST /step` — take an action
- `GET /state` — current network state
- `GET /tasks` — list all tasks

## Run Locally
```bash
pip install -r requirements.txt
python server.py
```

## Live Demo
https://huggingface.co/spaces/logicarius/trafficsolver