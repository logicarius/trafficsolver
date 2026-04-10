from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class RoadState(BaseModel):
    road_id: str
    from_node: str
    to_node: str
    capacity: int
    current_load: int
    utilization: float  # current_load / capacity


class NodeState(BaseModel):
    node_id: str
    queue_length: int
    is_bottleneck: bool


class TrafficObservation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    step: int
    max_steps: int
    nodes: List[NodeState]
    roads: List[RoadState]
    lci: float = Field(description="Load Concentration Index — lower is better, <0.4 is healthy")
    cdr: float = Field(description="Connector Dependency Ratio — lower is better")
    total_delay: float = Field(description="Sum of all queue lengths across all steps so far")
    average_utilization: float
    bottleneck_node: Optional[str] = None
    task_description: str
    hint: Optional[str] = None
    done: bool = False


class TrafficAction(BaseModel):
    """What the agent can do."""
    action_type: str = Field(
        description=(
            "One of: 'add_capacity', 'add_road', 'reroute', 'adjust_signal', 'do_nothing'"
        )
    )
    road_id: Optional[str] = Field(default=None, description="For add_capacity: which road to upgrade")
    capacity_increase: Optional[int] = Field(default=None, description="For add_capacity: how much to add (1-10)")
    from_node: Optional[str] = Field(default=None, description="For add_road: source node")
    to_node: Optional[str] = Field(default=None, description="For add_road: destination node")
    new_capacity: Optional[int] = Field(default=None, description="For add_road: capacity of new road (1-10)")
    target_node: Optional[str] = Field(default=None, description="For adjust_signal: which node")
    green_time: Optional[int] = Field(default=None, description="For adjust_signal: green time in seconds (10-60)")
    reasoning: Optional[str] = Field(default=None, description="Agent's reasoning for this action")


class TrafficReward(BaseModel):
    """Reward breakdown — partial signals at every step."""
    total: float = Field(description="Final score 0.0 to 1.0")
    congestion_reduction: float = Field(description="Reward for reducing LCI")
    bottleneck_cleared: float = Field(description="Reward for clearing the main bottleneck")
    flow_improvement: float = Field(description="Reward for improving average utilization balance")
    stability_bonus: float = Field(description="Bonus for maintaining stable flow over time")
    penalty: float = Field(description="Penalty for invalid actions or making things worse")
    breakdown: Dict[str, Any] = Field(default_factory=dict)
