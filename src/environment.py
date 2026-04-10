"""
environment.py — Core UrbanFlow OpenEnv environment.
Implements reset() / step() / state() API.
"""

from typing import Any, Dict, Optional, Tuple
from src.models import TrafficObservation, TrafficAction, TrafficReward, RoadState, NodeState
from src.network import TrafficNetwork, build_weak_network, build_resilient_network, build_dynamic_network
from src.routing import (
    compute_lci, compute_cdr, compute_total_delay,
    compute_average_utilization, find_bottleneck, simulate_one_step
)
from src.grader import grade_task1, grade_task2, grade_task3


TASKS = {
    "clear_bottleneck": {
        "id": "clear_bottleneck",
        "difficulty": "easy",
        "max_steps": 5,
        "description": (
            "A weak traffic network has all lanes merging into a single Connector node, "
            "causing severe congestion. Your job: reduce the queue at the main bottleneck "
            "by adding capacity or rerouting traffic. Target: reduce bottleneck queue by 50%."
        ),
        "hint": "The Connector node has queue=8. Add capacity to road r4 (Connector→MainRoad) or add a new parallel road.",
        "network_builder": build_weak_network,
    },
    "rebalance_network": {
        "id": "rebalance_network",
        "difficulty": "medium",
        "max_steps": 7,
        "description": (
            "A partially improved network still has uneven load distribution. "
            "LCI is above 0.4 (unhealthy) and CDR is too high (connector dependency). "
            "Redesign routing and add redundant paths to get LCI < 0.4 AND CDR < 0.7."
        ),
        "hint": None,
        "network_builder": build_resilient_network,
    },
    "maintain_stability": {
        "id": "maintain_stability",
        "difficulty": "hard",
        "max_steps": 10,
        "description": (
            "The network receives constant inflow every step. Without intervention, "
            "the Bridge node accumulates +15 vehicles per step — unbounded growth. "
            "Keep total delay under 600 across all 10 steps. Manage signals, "
            "add capacity, and reroute dynamically."
        ),
        "hint": None,
        "network_builder": build_dynamic_network,
    },
}


class UrbanFlowEnvironment:
    def __init__(self):
        self.network: Optional[TrafficNetwork] = None
        self.current_task_id: Optional[str] = None
        self.step_count: int = 0
        self.done: bool = False
        self.penalty_accumulator: float = 0.0
        self.delay_history: list[float] = []
        self._initial_bottleneck_queue: int = 0
        self._initial_lci: float = 0.0
        self._initial_cdr: float = 0.0
        self._last_reward: float = 0.0

    def reset(self, task_id: str = "clear_bottleneck") -> TrafficObservation:
        """Start a new episode for the given task."""
        if task_id not in TASKS:
            task_id = "clear_bottleneck"

        task = TASKS[task_id]
        self.network = task["network_builder"]()
        self.current_task_id = task_id
        self.step_count = 0
        self.done = False
        self.penalty_accumulator = 0.0
        self.delay_history = []

        # Record initial metrics
        self._initial_bottleneck_queue = max(
            (n.queue_length for n in self.network.nodes.values()), default=0
        )
        self._initial_lci = compute_lci(self.network)
        self._initial_cdr = compute_cdr(self.network)
        self._last_reward = 0.0

        return self._build_observation(task)

    def step(self, action: TrafficAction) -> Tuple[TrafficObservation, TrafficReward, bool, Dict]:
        """Apply agent action, simulate one step, return new observation + reward."""
        if self.done or self.network is None:
            raise RuntimeError("Call reset() before step()")

        task = TASKS[self.current_task_id]
        penalty = 0.0
        info = {"action_valid": True, "action_error": None}

        # --- Apply action ---
        valid, error = self._apply_action(action)
        if not valid:
            penalty = 0.1
            self.penalty_accumulator += penalty
            info["action_valid"] = False
            info["action_error"] = error

        # --- Simulate network dynamics for dynamic task ---
        if self.current_task_id == "maintain_stability":
            self.network = simulate_one_step(self.network, inflow=3)

        # --- Record delay ---
        delay = compute_total_delay(self.network)
        self.delay_history.append(delay)

        self.step_count += 1
        self.done = self.step_count >= task["max_steps"]

        # --- Compute reward ---
        reward = self._compute_reward(penalty)
        self._last_reward = reward.total

        obs = self._build_observation(task)
        obs.done = self.done

        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return full current state as a dict (for inspection/debugging)."""
        if self.network is None:
            return {"status": "not_initialized"}
        return {
            "task_id": self.current_task_id,
            "step": self.step_count,
            "done": self.done,
            "lci": compute_lci(self.network),
            "cdr": compute_cdr(self.network),
            "total_delay": compute_total_delay(self.network),
            "delay_history": self.delay_history,
            "nodes": {nid: n.queue_length for nid, n in self.network.nodes.items()},
            "roads": {
                rid: {"load": r.current_load, "capacity": r.capacity, "util": round(r.utilization, 3)}
                for rid, r in self.network.roads.items()
            },
            "penalty_accumulator": self.penalty_accumulator,
            "last_reward": self._last_reward,
        }

    def list_tasks(self) -> list:
        return [
            {
                "task_id": t["id"],
                "difficulty": t["difficulty"],
                "max_steps": t["max_steps"],
                "description": t["description"],
            }
            for t in TASKS.values()
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_action(self, action: TrafficAction) -> Tuple[bool, Optional[str]]:
        """Mutate the network based on the agent's action. Returns (valid, error)."""
        net = self.network

        if action.action_type == "do_nothing":
            return True, None

        elif action.action_type == "add_capacity":
            if action.road_id not in net.roads:
                return False, f"Road '{action.road_id}' does not exist"
            increase = min(max(action.capacity_increase or 2, 1), 10)
            net.roads[action.road_id].capacity += increase
            return True, None

        elif action.action_type == "add_road":
            if not action.from_node or not action.to_node:
                return False, "add_road requires from_node and to_node"
            if action.from_node not in net.nodes or action.to_node not in net.nodes:
                return False, f"Node '{action.from_node}' or '{action.to_node}' does not exist"
            cap = min(max(action.new_capacity or 4, 1), 10)
            new_id = f"r_new_{len(net.roads)}"
            net.add_road(new_id, action.from_node, action.to_node, capacity=cap, load=0)
            return True, None

        elif action.action_type == "reroute":
            # Shift load from most congested road to new road if parallel exists
            congested = sorted(net.roads.values(), key=lambda r: r.utilization, reverse=True)
            if congested:
                road = congested[0]
                shift = max(1, road.current_load // 3)
                road.current_load = max(0, road.current_load - shift)
                # Update downstream node queue
                if road.to_node in net.nodes:
                    net.nodes[road.to_node].queue_length = max(
                        0, net.nodes[road.to_node].queue_length - shift
                    )
            return True, None

        elif action.action_type == "adjust_signal":
            if not action.target_node or action.target_node not in net.nodes:
                return False, f"Node '{action.target_node}' does not exist"
            green = min(max(action.green_time or 30, 10), 60)
            net.nodes[action.target_node].signal_green_time = green
            # More green time = more throughput = reduce queue
            extra_throughput = (green - 30) // 10
            net.nodes[action.target_node].queue_length = max(
                0, net.nodes[action.target_node].queue_length - extra_throughput
            )
            return True, None

        else:
            return False, f"Unknown action_type '{action.action_type}'"

    def _compute_reward(self, penalty: float) -> TrafficReward:
        """Compute reward based on current task."""
        task_id = self.current_task_id
        net = self.network

        if task_id == "clear_bottleneck":
            result = grade_task1(
                net, self._initial_bottleneck_queue,
                self.step_count, TASKS[task_id]["max_steps"]
            )
        elif task_id == "rebalance_network":
            result = grade_task2(net, self._initial_lci, self._initial_cdr, self.step_count)
        else:  # maintain_stability
            result = grade_task3(self.delay_history)

        score = max(0.0, result["score"] - self.penalty_accumulator)
        breakdown = result.get("breakdown", {})

        return TrafficReward(
            total=round(score, 3),
            congestion_reduction=breakdown.get("congestion_reduction", 0.0),
            bottleneck_cleared=breakdown.get("lci_bonus", breakdown.get("lci_target_met", 0.0)),
            flow_improvement=breakdown.get("cdr_target_met", breakdown.get("final_delay_target", 0.0)),
            stability_bonus=breakdown.get("stability_maintained", breakdown.get("speed_bonus", 0.0)),
            penalty=round(self.penalty_accumulator, 3),
            breakdown=breakdown,
        )

    def _build_observation(self, task: dict) -> TrafficObservation:
        net = self.network
        nodes = [
            NodeState(
                node_id=nid,
                queue_length=n.queue_length,
                is_bottleneck=n.is_bottleneck,
            )
            for nid, n in net.nodes.items()
        ]
        roads = [
            RoadState(
                road_id=rid,
                from_node=r.from_node,
                to_node=r.to_node,
                capacity=r.capacity,
                current_load=r.current_load,
                utilization=round(r.utilization, 3),
            )
            for rid, r in net.roads.items()
        ]
        return TrafficObservation(
            task_id=self.current_task_id,
            step=self.step_count,
            max_steps=task["max_steps"],
            nodes=nodes,
            roads=roads,
            lci=compute_lci(net),
            cdr=compute_cdr(net),
            total_delay=compute_total_delay(net),
            average_utilization=compute_average_utilization(net),
            bottleneck_node=find_bottleneck(net),
            task_description=task["description"],
            hint=task.get("hint") if task["difficulty"] == "easy" else None,
            done=self.done,
        )
