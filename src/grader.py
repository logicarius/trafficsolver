"""
grader.py — Deterministic scoring for all 3 UrbanFlow tasks.
Every grader returns a float in [0.0, 1.0].
"""

from src.network import TrafficNetwork
from src.routing import compute_lci, compute_cdr, compute_total_delay, find_bottleneck


# ── Task 1: Clear the Bottleneck ─────────────────────────────────────────────

def grade_task1(
    network: TrafficNetwork,
    initial_bottleneck_queue: int,
    steps_taken: int,
    max_steps: int,
) -> dict:
    """
    Easy task: Reduce the queue at the main bottleneck node.

    Scoring:
      +0.4  bottleneck queue reduced by ≥50%
      +0.3  bottleneck queue reduced by ≥25%
      +0.2  any reduction in bottleneck queue
      +0.1  speed bonus for solving in ≤3 steps
      -0.1  penalty per invalid action (tracked externally)
    """
    bottleneck_id = find_bottleneck(network)
    if bottleneck_id is None:
        return {"score": 0.0, "reason": "No bottleneck found"}

    current_queue = network.nodes[bottleneck_id].queue_length
    if initial_bottleneck_queue == 0:
        reduction_ratio = 1.0
    else:
        reduction_ratio = max(0.0, 1.0 - current_queue / initial_bottleneck_queue)

    score = 0.0
    breakdown = {}

    if reduction_ratio >= 0.5:
        score += 0.4
        breakdown["congestion_reduction"] = 0.4
    elif reduction_ratio >= 0.25:
        score += 0.3
        breakdown["congestion_reduction"] = 0.3
    elif reduction_ratio > 0.0:
        score += 0.2
        breakdown["congestion_reduction"] = 0.2
    else:
        breakdown["congestion_reduction"] = 0.0

    # LCI improvement bonus
    lci = compute_lci(network)
    if lci < 0.3:
        score += 0.3
        breakdown["lci_bonus"] = 0.3
    elif lci < 0.4:
        score += 0.2
        breakdown["lci_bonus"] = 0.2
    else:
        breakdown["lci_bonus"] = 0.0

    # Speed bonus
    if steps_taken <= 3 and score > 0.3:
        score += 0.1
        breakdown["speed_bonus"] = 0.1
    else:
        breakdown["speed_bonus"] = 0.0

    breakdown["reduction_ratio"] = round(reduction_ratio, 3)
    breakdown["current_queue"] = current_queue
    breakdown["lci"] = lci

    return {"score": round(min(score, 1.0), 3), "breakdown": breakdown}


# ── Task 2: Rebalance the Network ────────────────────────────────────────────

def grade_task2(
    network: TrafficNetwork,
    initial_lci: float,
    initial_cdr: float,
    steps_taken: int,
) -> dict:
    """
    Medium task: Reduce LCI below 0.4 AND CDR below 0.7.

    Scoring:
      +0.35  LCI < 0.4 (healthy distribution)
      +0.25  CDR < 0.7 (reduced connector dependency)
      +0.25  both conditions met simultaneously
      +0.15  LCI improved from initial (partial credit)
    """
    lci = compute_lci(network)
    cdr = compute_cdr(network)

    score = 0.0
    breakdown = {}

    # LCI target
    if lci < 0.4:
        score += 0.35
        breakdown["lci_target_met"] = 0.35
    elif lci < initial_lci:
        improvement = (initial_lci - lci) / initial_lci
        partial = 0.15 * improvement
        score += partial
        breakdown["lci_partial"] = round(partial, 3)
    else:
        breakdown["lci_target_met"] = 0.0

    # CDR target
    if cdr < 0.7:
        score += 0.25
        breakdown["cdr_target_met"] = 0.25
    elif cdr < initial_cdr:
        improvement = (initial_cdr - cdr) / initial_cdr
        partial = 0.1 * improvement
        score += partial
        breakdown["cdr_partial"] = round(partial, 3)
    else:
        breakdown["cdr_target_met"] = 0.0

    # Both targets met simultaneously
    if lci < 0.4 and cdr < 0.7:
        score += 0.25
        breakdown["both_targets_bonus"] = 0.25
    else:
        breakdown["both_targets_bonus"] = 0.0

    breakdown["final_lci"] = lci
    breakdown["final_cdr"] = cdr
    breakdown["initial_lci"] = initial_lci
    breakdown["initial_cdr"] = initial_cdr

    return {"score": round(min(score, 1.0), 3), "breakdown": breakdown}


# ── Task 3: Maintain Dynamic Stability ───────────────────────────────────────

def grade_task3(
    delay_history: list[float],
    max_allowed_delay: float = 600.0,
    target_delay: float = 300.0,
) -> dict:
    """
    Hard task: Keep total delay under control across 10 simulation steps.
    The network has constant inflow — without intervention, delay grows unbounded.

    Scoring:
      +0.4   final delay < target_delay (300)
      +0.3   delay never exceeded max_allowed_delay at any step
      +0.2   delay trend is decreasing (agent actively improving)
      +0.1   delay at step 10 < delay at step 1 (net improvement)
    """
    if not delay_history:
        return {"score": 0.0, "reason": "No delay history"}

    final_delay = delay_history[-1]
    max_delay = max(delay_history)
    initial_delay = delay_history[0]

    score = 0.0
    breakdown = {}

    # Final delay target
    if final_delay < target_delay:
        score += 0.4
        breakdown["final_delay_target"] = 0.4
    elif final_delay < max_allowed_delay:
        partial = 0.2 * (1 - final_delay / max_allowed_delay)
        score += partial
        breakdown["final_delay_partial"] = round(partial, 3)
    else:
        breakdown["final_delay_target"] = 0.0

    # Never exceeded max
    if max_delay < max_allowed_delay:
        score += 0.3
        breakdown["stability_maintained"] = 0.3
    else:
        breakdown["stability_maintained"] = 0.0

    # Decreasing trend (last half better than first half)
    if len(delay_history) >= 4:
        mid = len(delay_history) // 2
        first_half_avg = sum(delay_history[:mid]) / mid
        second_half_avg = sum(delay_history[mid:]) / (len(delay_history) - mid)
        if second_half_avg < first_half_avg:
            score += 0.2
            breakdown["decreasing_trend"] = 0.2
        else:
            breakdown["decreasing_trend"] = 0.0

    # Net improvement
    if final_delay < initial_delay:
        score += 0.1
        breakdown["net_improvement"] = 0.1
    else:
        breakdown["net_improvement"] = 0.0

    breakdown["final_delay"] = final_delay
    breakdown["max_delay"] = max_delay
    breakdown["initial_delay"] = initial_delay
    breakdown["delay_history"] = [round(d, 1) for d in delay_history]

    return {"score": round(min(score, 1.0), 3), "breakdown": breakdown}
