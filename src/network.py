"""
network.py — Traffic network definitions for UrbanFlow OpenEnv
Builds weak (bottleneck) and resilient (redundant) network topologies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import copy


@dataclass
class Road:
    road_id: str
    from_node: str
    to_node: str
    capacity: int
    current_load: int = 0

    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 1.0
        return min(self.current_load / self.capacity, 1.0)

    @property
    def is_congested(self) -> bool:
        return self.utilization >= 0.8


@dataclass
class Node:
    node_id: str
    queue_length: int = 0
    signal_green_time: int = 30  # seconds

    @property
    def is_bottleneck(self) -> bool:
        return self.queue_length > 5


@dataclass
class TrafficNetwork:
    nodes: Dict[str, Node] = field(default_factory=dict)
    roads: Dict[str, Road] = field(default_factory=dict)

    def add_node(self, node_id: str, queue: int = 0) -> None:
        self.nodes[node_id] = Node(node_id, queue)

    def add_road(self, road_id: str, from_node: str, to_node: str, capacity: int, load: int = 0) -> None:
        self.roads[road_id] = Road(road_id, from_node, to_node, capacity, load)

    def get_roads_from(self, node_id: str) -> List[Road]:
        return [r for r in self.roads.values() if r.from_node == node_id]

    def get_roads_to(self, node_id: str) -> List[Road]:
        return [r for r in self.roads.values() if r.to_node == node_id]

    def clone(self) -> "TrafficNetwork":
        return copy.deepcopy(self)


def build_weak_network() -> TrafficNetwork:
    """
    Stage 1: Weak topology — all lanes funnel into one Connector node.
    Guaranteed structural congestion regardless of demand volume.

    Topology:
        Lane1 ──┐
        Lane2 ──┼──► Connector ──► MainRoad ──► Exit
        Lane3 ──┘
    """
    net = TrafficNetwork()
    net.add_node("Lane1", queue=3)
    net.add_node("Lane2", queue=4)
    net.add_node("Lane3", queue=3)
    net.add_node("Connector", queue=8)   # bottleneck
    net.add_node("MainRoad", queue=2)
    net.add_node("Exit", queue=0)

    net.add_road("r1", "Lane1",     "Connector", capacity=4, load=4)
    net.add_road("r2", "Lane2",     "Connector", capacity=4, load=4)
    net.add_road("r3", "Lane3",     "Connector", capacity=4, load=4)
    net.add_road("r4", "Connector", "MainRoad",  capacity=5, load=5)
    net.add_road("r5", "MainRoad",  "Exit",      capacity=8, load=3)
    return net


def build_resilient_network() -> TrafficNetwork:
    """
    Stage 2: Resilient topology — multiple paths, redundancy added.
    LCI should be below 0.4 after agent actions.

    Topology:
        Lane1 ──► MainRoad ──► Exit
        Lane2 ──► Connector ──► MainRoad
        Lane3 ──► SideRoad ──► Exit
                  Connector ──► SideRoad (bypass)
    """
    net = TrafficNetwork()
    net.add_node("Lane1",     queue=2)
    net.add_node("Lane2",     queue=3)
    net.add_node("Lane3",     queue=2)
    net.add_node("Connector", queue=5)
    net.add_node("MainRoad",  queue=3)
    net.add_node("SideRoad",  queue=2)
    net.add_node("Exit",      queue=0)

    net.add_road("r1", "Lane1",     "MainRoad",  capacity=5, load=3)
    net.add_road("r2", "Lane2",     "Connector", capacity=5, load=4)
    net.add_road("r3", "Lane3",     "SideRoad",  capacity=5, load=3)
    net.add_road("r4", "Connector", "MainRoad",  capacity=5, load=4)
    net.add_road("r5", "Connector", "SideRoad",  capacity=3, load=2)
    net.add_road("r6", "MainRoad",  "Exit",      capacity=8, load=5)
    net.add_road("r7", "SideRoad",  "Exit",      capacity=6, load=3)
    return net


def build_dynamic_network() -> TrafficNetwork:
    """
    Stage 3: Dynamic network — same resilient structure but with constant
    inflow. Agent must keep total delay stable over 10 time steps.
    Bridge node accumulates +15 vehicles per step without intervention.
    """
    net = build_resilient_network()
    # Add a Bridge node that accumulates over time
    net.add_node("Bridge", queue=4)
    net.add_road("r8", "MainRoad", "Bridge", capacity=4, load=4)
    net.add_road("r9", "Bridge",   "Exit",   capacity=3, load=2)
    return net
