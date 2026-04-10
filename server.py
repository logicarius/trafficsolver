"""
server.py — FastAPI server exposing UrbanFlow as an OpenEnv HTTP API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from src.environment import UrbanFlowEnvironment
from src.models import TrafficAction

app = FastAPI(
    title="UrbanFlow OpenEnv",
    description="Traffic signal optimization environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = UrbanFlowEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "clear_bottleneck"


@app.get("/health")
def health():
    return {"status": "ok", "env": "urbanflow"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    task_id = (request.task_id if request else None) or "clear_bottleneck"
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
def step(action: TrafficAction):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    return env.list_tasks()


@app.get("/")
def root():
    return {
        "name": "UrbanFlow",
        "description": "Urban traffic congestion management environment for AI agents.",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
        "tasks": ["clear_bottleneck (easy)", "rebalance_network (medium)", "maintain_stability (hard)"],
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
