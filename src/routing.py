"""
routing.py — Greedy capacity-based routing + congestion metrics
"""

from typing import Dict, List, Tuple
from src.network import TrafficNetwork, Road


def compute_lci(network: TrafficNetwork) -> float:
    """
    Load Concentration Index — measures how unevenly load is distributed.
    0.0 = perfectly balanced. >0.4 = unhealthy concentration.
    """
    utilizations = [r.utilization for r in network.roads.values()]
    if not utilizations:
        return 0.0
    mean_util = sum(utilizations) / len(utilizations)
    variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
    return round(variance ** 0.5, 4)  # std deviation as concentration index


def compute_cdr(network: TrafficNetwork, connector_node: str = "Connector") -> float:
    """
    Connector Dependency Ratio — fraction of all traffic passing through
    the connector node. 1.0 = all traffic depends on it (worst case).
    """
    total_load = sum(r.current_load for r in network.roads.values())
    if total_load == 0:
        return 0.0
    connector_load = sum(
        r.current_load for r in network.roads.values()
        if r.from_node == connector_node or r.to_node == connector_node
    )
    return round(min(connector_load / total_load, 1.0), 4)


def compute_total_delay(network: TrafficNetwork) -> float:
    """Total queue length across all nodes — lower is better."""
    return float(sum(n.queue_length for n in network.nodes.values()))


def compute_average_utilization(network: TrafficNetwork) -> float:
    utils = [r.utilization for r in network.roads.values()]
    return round(sum(utils) / len(utils), 4) if utils else 0.0


def find_bottleneck(network: TrafficNetwork) -> str | None:
    """Returns the node with the highest queue length."""
    if not network.nodes:
        return None
    return max(network.nodes.values(), key=lambda n: n.queue_length).node_id


def apply_greedy_routing(network: TrafficNetwork) -> TrafficNetwork:
    """
    Redistribute load from congested roads to less-used parallel roads.
    This is what the agent is trying to help with.
    """
    for node_id, node in network.nodes.items():
        outgoing = network.get_roads_from(node_id)
        if len(outgoing) <= 1:
            continue
        # Sort by utilization ascending — prefer less loaded roads
        outgoing.sort(key=lambda r: r.utilization)
        total_load = sum(r.current_load for r in outgoing)
        # Redistribute proportionally to available capacity
        total_cap = sum(r.capacity for r in outgoing)
        for road in outgoing:
            share = road.capacity / total_cap if total_cap > 0 else 1 / len(outgoing)
            road.current_load = int(total_load * share)
    return network


def simulate_one_step(network: TrafficNetwork, inflow: int = 3) -> TrafficNetwork:
    """
    Advance the simulation by one time step.
    Each node receives inflow, processes what it can, accumulates queue.
    """
    for node_id, node in network.nodes.items():
        incoming_roads = network.get_roads_to(node_id)
        outgoing_roads = network.get_roads_from(node_id)

        # Total incoming load
        incoming = sum(r.current_load for r in incoming_roads)
        if node_id in ["Lane1", "Lane2", "Lane3"]:
            incoming += inflow  # source nodes get fresh traffic

        # Total outgoing capacity
        outgoing_cap = sum(r.capacity for r in outgoing_roads)

        if node_id == "Exit":
            node.queue_length = 0  # exit always drains
            continue

        processed = min(incoming + node.queue_length, outgoing_cap)
        node.queue_length = max(0, incoming + node.queue_length - processed)

    return network
