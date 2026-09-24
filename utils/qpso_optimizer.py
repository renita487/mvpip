"""
========================================================================================
EGREEN QUANTA — QUANTUM-INSPIRED PARTICLE SWARM OPTIMIZATION (QPSO) ENGINE
& CLASSICAL PSO BENCHMARKING SUITE
========================================================================================
Implements:
  1. Quantum-behaved Particle Swarm Optimization (QPSO) based on delta potential-well wave mechanics.
  2. Mixed-integer decision space handling (vessel assignment, cargo allocation, continuous speed, fuel selection).
  3. Constrained Multi-Objective Optimization (Fuel, Operational Cost, Lifecycle GHG CO2e).
  4. Non-dominated Pareto frontier extraction (Plan A Eco, Plan B Balanced, Plan C Fast).
  5. Empirical Classical PSO vs. QPSO benchmarking with authentic convergence and runtime telemetry.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from utils.physics_engine import calculate_hydrodynamic_fuel, VESSEL_SPECS
from utils.fuel_scenarios import calculate_lifecycle_emissions, get_fuel_spec, FUEL_DATABASE
from utils.constraint_engine import default_constraint_engine, FleetConstraintEngine

# Default fleet catalog for multi-vessel routing
DEFAULT_FLEET: List[Dict[str, Any]] = [
    {
        "id": "V01",
        "name": "Pacific Voyager",
        "type": "Container Ship",
        "dwt": 85000.0,
        "min_speed": 10.0,
        "max_speed": 22.0,
        "engine_power_kw": 48000.0,
        "status": "Active",
        "available": True,
        "fuel_compatibility": ["HFO", "VLSFO", "MGO", "LNG", "Methanol"],
        "has_shore_power": True,
        "icon": "🚢"
    },
    {
        "id": "V02",
        "name": "Nordic Bulk Carrier",
        "type": "Bulk Carrier",
        "dwt": 120000.0,
        "min_speed": 9.0,
        "max_speed": 16.0,
        "engine_power_kw": 16500.0,
        "status": "Active",
        "available": True,
        "fuel_compatibility": ["HFO", "VLSFO", "MGO", "Ammonia"],
        "has_shore_power": True,
        "icon": "⛴️"
    },
    {
        "id": "V03",
        "name": "Atlantic Titan",
        "type": "Tanker",
        "dwt": 160000.0,
        "min_speed": 9.5,
        "max_speed": 17.0,
        "engine_power_kw": 22000.0,
        "status": "Active",
        "available": True,
        "fuel_compatibility": ["HFO", "VLSFO", "MGO", "LNG"],
        "has_shore_power": False,
        "icon": "🛢️"
    },
    {
        "id": "V04",
        "name": "Global Feeder 04",
        "type": "General Cargo",
        "dwt": 35000.0,
        "min_speed": 8.5,
        "max_speed": 15.0,
        "engine_power_kw": 9500.0,
        "status": "Active",
        "available": True,
        "fuel_compatibility": ["VLSFO", "MGO", "Methanol", "Hydrogen"],
        "has_shore_power": True,
        "icon": "📦"
    },
    {
        "id": "V05",
        "name": "Emerald Express",
        "type": "Container Ship",
        "dwt": 85000.0,
        "min_speed": 10.0,
        "max_speed": 22.0,
        "engine_power_kw": 48000.0,
        "status": "Maintenance",
        "available": False,
        "fuel_compatibility": ["HFO", "VLSFO", "MGO", "LNG"],
        "has_shore_power": True,
        "icon": "🚢"
    }
]


def decode_particle_to_fleet_plan(
    particle_vector: np.ndarray,
    fleet_vessels: List[Dict[str, Any]],
    cargo_demand_tonnes: float,
    port_available_fuels: List[str]
) -> Tuple[List[bool], List[float], List[float], List[str]]:
    """
    Mixed-Integer Decision Space Decoder:
    Maps continuous particle coordinates in R^(4N) to:
      1. active_flags: bool per vessel (activation)
      2. cargo_allocations: tonnes per vessel (repaired to meet demand <= capacity)
      3. speeds: knots per vessel (clamped to [min_speed, max_speed])
      4. fuel_choices: fuel string per vessel (mapped to valid compatible fuels)
    """
    n = len(fleet_vessels)
    raw_active_scores = []
    raw_cargo_weights = []
    speeds = []
    fuel_choices = []

    for i, ship in enumerate(fleet_vessels):
        idx = i * 4
        act_val = particle_vector[idx]
        cg_val = max(0.001, particle_vector[idx + 1])
        sp_val = particle_vector[idx + 2]
        fl_val = particle_vector[idx + 3]

        min_sp = ship.get("min_speed", 10.0)
        max_sp = ship.get("max_speed", 22.0)
        clamped_sp = float(np.clip(sp_val, min_sp, max_sp))
        speeds.append(round(clamped_sp, 1))

        # Fuel choice mapping
        compat = ship.get("fuel_compatibility", ["HFO", "VLSFO", "MGO"])
        # Intersection with port available fuels if possible
        valid_fuels = [f for f in compat if f in port_available_fuels]
        if not valid_fuels:
            valid_fuels = compat

        fuel_idx = int(np.clip(np.floor(fl_val), 0, len(valid_fuels) - 1))
        fuel_choices.append(valid_fuels[fuel_idx])

        # Active status
        is_avail = ship.get("status", "Active") == "Active" and ship.get("available", True)
        if is_avail and act_val >= 0.40:
            raw_active_scores.append(True)
        else:
            raw_active_scores.append(False)

        raw_cargo_weights.append(cg_val)

    # Ensure at least one available vessel is active if cargo demand > 0
    avail_indices = [i for i, s in enumerate(fleet_vessels) if s.get("status", "Active") == "Active" and s.get("available", True)]
    active_flags = list(raw_active_scores)
    
    if not any(active_flags[i] for i in avail_indices) and avail_indices:
        # Activate vessel with largest capacity
        best_idx = max(avail_indices, key=lambda i: fleet_vessels[i].get("dwt", 50000.0))
        active_flags[best_idx] = True

    # Cargo allocation with capacity clipping & normalization
    cargo_allocations = [0.0] * n
    active_indices = [i for i, act in enumerate(active_flags) if act]
    
    if active_indices:
        total_active_cap = sum(fleet_vessels[i].get("dwt", 80000.0) for i in active_indices)
        # Proportionate distribution based on weights * capacity
        weighted_caps = [raw_cargo_weights[i] * fleet_vessels[i].get("dwt", 80000.0) for i in active_indices]
        sum_weights = sum(weighted_caps) if sum(weighted_caps) > 0 else 1.0

        for i, idx in enumerate(active_indices):
            cap = fleet_vessels[idx].get("dwt", 80000.0)
            share = (weighted_caps[i] / sum_weights) * cargo_demand_tonnes
            alloc = min(cap, share)
            cargo_allocations[idx] = round(float(alloc), 1)

        # Second pass adjustment to cover any rounding gap if within capacity
        curr_total = sum(cargo_allocations)
        gap = cargo_demand_tonnes - curr_total
        if gap > 0:
            for idx in active_indices:
                room = fleet_vessels[idx].get("dwt", 80000.0) - cargo_allocations[idx]
                add = min(room, gap)
                cargo_allocations[idx] += add
                gap -= add
                if gap <= 0.1:
                    break

    return active_flags, cargo_allocations, speeds, fuel_choices


def evaluate_fleet_plan(
    fleet_vessels: List[Dict[str, Any]],
    active_flags: List[bool],
    cargo_allocations: List[float],
    speeds: List[float],
    fuel_choices: List[str],
    distance_nm: float,
    weather_condition: str,
    cargo_demand_tonnes: float,
    schedule_deadline_hours: float,
    port_available_fuels: List[str],
    charter_rate_per_hour: float = 350.0,
    carbon_tax_rate: float = 85.0,
    include_carbon_tax: bool = False,
    require_shore_power: bool = False,
    constraint_engine: Optional[FleetConstraintEngine] = None,
    weight_fuel: float = 0.35,
    weight_cost: float = 0.35,
    weight_ghg: float = 0.30
) -> Dict[str, Any]:
    """
    Multi-Objective Evaluation Function:
    Calculates total fleet fuel, total operational cost ($), lifecycle Well-to-Wake GHG (t CO2e),
    max transit duration, constraint penalties, and composite scalar fitness.
    """
    c_engine = constraint_engine if constraint_engine else default_constraint_engine

    # 1. Evaluate operational constraints
    c_res = c_engine.evaluate_plan(
        fleet_vessels=fleet_vessels,
        cargo_allocations=cargo_allocations,
        speeds=speeds,
        fuel_choices=fuel_choices,
        active_flags=active_flags,
        voyage_distance_nm=distance_nm,
        cargo_demand_tonnes=cargo_demand_tonnes,
        schedule_deadline_hours=schedule_deadline_hours,
        port_available_fuels=port_available_fuels,
        require_shore_power=require_shore_power
    )

    total_fuel_tonnes = 0.0
    total_cost_usd = 0.0
    total_wtw_ghg_co2e = 0.0
    total_ttw_co2 = 0.0
    total_wtt_ghg = 0.0
    max_transit_time_hrs = 0.0
    vessel_breakdown = []

    for i, ship in enumerate(fleet_vessels):
        if not active_flags[i]:
            continue

        sp = speeds[i]
        cg = cargo_allocations[i]
        fl = fuel_choices[i]
        
        # Physics hydrodynamics
        phys = calculate_hydrodynamic_fuel(
            vessel_type=ship["type"],
            speed_knots=sp,
            cargo_tonnes=cg,
            weather_condition=weather_condition,
            distance_nm=distance_nm,
            fuel_type=fl,
            custom_specs=ship
        )

        # Emissions & costs
        em = calculate_lifecycle_emissions(
            fuel_type=fl,
            fuel_tonnes=phys["total_voyage_fuel_tonnes"],
            cargo_tonnes=cg,
            distance_nm=distance_nm,
            include_carbon_tax=include_carbon_tax,
            carbon_tax_rate=carbon_tax_rate,
            use_shore_power=(require_shore_power and ship.get("has_shore_power", True))
        )

        time_charter_cost = phys["voyage_hours"] * charter_rate_per_hour
        ship_total_cost = em["total_fuel_cost_usd"] + time_charter_cost

        total_fuel_tonnes += phys["total_voyage_fuel_tonnes"]
        total_cost_usd += ship_total_cost
        total_wtw_ghg_co2e += em["wtw_lifecycle_ghg_co2e_tonnes"]
        total_ttw_co2 += em["ttw_co2_tonnes"]
        total_wtt_ghg += em["wtt_ghg_co2e_tonnes"]
        
        if phys["voyage_hours"] > max_transit_time_hrs:
            max_transit_time_hrs = phys["voyage_hours"]

        vessel_breakdown.append({
            "vessel_id": ship.get("id", f"V{i+1}"),
            "vessel_name": ship["name"],
            "type": ship["type"],
            "icon": ship.get("icon", "🚢"),
            "allocated_cargo_tonnes": round(cg, 1),
            "speed_knots": round(sp, 1),
            "fuel_choice": fl,
            "fuel_consumed_tonnes": round(phys["total_voyage_fuel_tonnes"], 2),
            "fuel_per_day_tonnes": round(phys["fuel_per_day_tonnes"], 2),
            "ttw_co2_tonnes": round(em["ttw_co2_tonnes"], 2),
            "wtt_ghg_tonnes": round(em["wtt_ghg_co2e_tonnes"], 2),
            "wtw_lifecycle_co2e_tonnes": round(em["wtw_lifecycle_ghg_co2e_tonnes"], 2),
            "bunker_cost_usd": round(em["direct_bunker_cost_usd"], 2),
            "time_charter_cost_usd": round(time_charter_cost, 2),
            "total_ship_cost_usd": round(ship_total_cost, 2),
            "transit_hours": round(phys["voyage_hours"], 1),
            "cii_rating": em["cii_rating"],
            "cii_badge": em["cii_badge"],
            "cii_color": em["cii_color"]
        })

    # Normalization scaling factors for multi-objective aggregation
    scale_fuel = max(10.0, cargo_demand_tonnes * 0.005 * (distance_nm / 1000.0))
    scale_cost = max(5000.0, scale_fuel * 650.0 + 5000.0)
    scale_ghg = max(30.0, scale_fuel * 3.17)

    norm_fuel = total_fuel_tonnes / scale_fuel
    norm_cost = total_cost_usd / scale_cost
    norm_ghg = total_wtw_ghg_co2e / scale_ghg

    composite_objective = (
        (weight_fuel * norm_fuel) +
        (weight_cost * norm_cost) +
        (weight_ghg * norm_ghg)
    )

    # Add constraint penalty
    fitness_value = composite_objective + (c_res["penalty_score"] / 1000.0)

    return {
        "fitness": round(float(fitness_value), 4),
        "composite_objective": round(float(composite_objective), 4),
        "is_feasible": c_res["is_feasible"],
        "violations": c_res["violations"],
        "penalty_score": c_res["penalty_score"],
        "total_fuel_tonnes": round(total_fuel_tonnes, 2),
        "total_cost_usd": round(total_cost_usd, 2),
        "total_wtw_ghg_co2e": round(total_wtw_ghg_co2e, 2),
        "total_ttw_co2_tonnes": round(total_ttw_co2, 2),
        "total_wtt_ghg_tonnes": round(total_wtt_ghg, 2),
        "max_transit_time_hrs": round(max_transit_time_hrs, 1),
        "active_vessels_count": c_res["num_active_vessels"],
        "total_allocated_cargo_tonnes": c_res["total_allocated_cargo_tonnes"],
        "vessel_breakdown": vessel_breakdown
    }


def run_qpso_optimization(
    fleet_vessels: List[Dict[str, Any]],
    cargo_demand_tonnes: float,
    distance_nm: float,
    weather_condition: str = "Moderate",
    schedule_deadline_hours: float = 120.0,
    port_available_fuels: Optional[List[str]] = None,
    charter_rate_per_hour: float = 350.0,
    carbon_tax_rate: float = 85.0,
    include_carbon_tax: bool = False,
    require_shore_power: bool = False,
    swarm_size: int = 30,
    max_iterations: int = 60,
    weight_fuel: float = 0.35,
    weight_cost: float = 0.35,
    weight_ghg: float = 0.30,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Quantum-behaved Particle Swarm Optimization (QPSO) Algorithm.
    Searches multi-vessel fleet configuration using quantum delta potential well dynamics.
    Tracks convergence, runtime, and extracts non-dominated Pareto frontier solutions.
    """
    start_time = time.perf_counter()
    np.random.seed(random_seed)

    fuels = port_available_fuels if port_available_fuels else list(FUEL_DATABASE.keys())
    n = len(fleet_vessels)
    dim = n * 4

    # Dimension bounds [lower, upper]
    bounds_low = np.zeros(dim)
    bounds_high = np.zeros(dim)
    for i, ship in enumerate(fleet_vessels):
        idx = i * 4
        bounds_low[idx] = 0.0       # Active prob [0, 1]
        bounds_high[idx] = 1.0
        bounds_low[idx+1] = 0.05    # Cargo weight [0.05, 1.0]
        bounds_high[idx+1] = 1.0
        bounds_low[idx+2] = ship.get("min_speed", 10.0)  # Speed [min, max]
        bounds_high[idx+2] = ship.get("max_speed", 22.0)
        bounds_low[idx+3] = 0.0     # Fuel index
        compat = ship.get("fuel_compatibility", ["HFO", "VLSFO", "MGO"])
        bounds_high[idx+3] = max(1.0, float(len(compat)))

    # Initialize Swarm Positions X in R^(M x D)
    X = np.random.uniform(bounds_low, bounds_high, size=(swarm_size, dim))
    pbest = np.copy(X)
    pbest_fitness = np.full(swarm_size, np.inf)
    pbest_evals = [None] * swarm_size

    gbest = np.copy(X[0])
    gbest_fitness = np.inf
    gbest_eval = None

    convergence_history = []
    all_evaluated_solutions = []

    # Initial evaluation
    for j in range(swarm_size):
        act, c_alloc, spd, f_choice = decode_particle_to_fleet_plan(X[j], fleet_vessels, cargo_demand_tonnes, fuels)
        ev = evaluate_fleet_plan(
            fleet_vessels, act, c_alloc, spd, f_choice, distance_nm, weather_condition,
            cargo_demand_tonnes, schedule_deadline_hours, fuels,
            charter_rate_per_hour, carbon_tax_rate, include_carbon_tax, require_shore_power,
            weight_fuel=weight_fuel, weight_cost=weight_cost, weight_ghg=weight_ghg
        )
        pbest_fitness[j] = ev["fitness"]
        pbest_evals[j] = ev

        if ev["is_feasible"]:
            all_evaluated_solutions.append(ev)

        if ev["fitness"] < gbest_fitness:
            gbest_fitness = ev["fitness"]
            gbest = np.copy(X[j])
            gbest_eval = ev

    # Fallback if no feasible found initially
    if gbest_eval is None:
        gbest_eval = pbest_evals[0]

    # QPSO Iterative Search Loop
    beta_max = 1.0
    beta_min = 0.5

    for t in range(max_iterations):
        # Adaptive contraction-expansion coefficient beta
        beta = beta_max - (beta_max - beta_min) * (t / max_iterations)

        # 1. Compute Mean Best Position of the swarm (mbest)
        mbest = np.mean(pbest, axis=0)

        for j in range(swarm_size):
            # 2. Local Stochastic Attractor p_jd = phi * pbest_jd + (1 - phi) * gbest_d
            phi = np.random.uniform(0.0, 1.0, size=dim)
            p_local = phi * pbest[j] + (1.0 - phi) * gbest

            # 3. Quantum Delta Potential Position Update
            u = np.random.uniform(0.0, 1.0, size=dim)
            # Avoid ln(0)
            u = np.maximum(u, 1e-7)
            
            # Quantum wave function perturbation
            quantum_step = beta * np.abs(mbest - X[j]) * np.log(1.0 / u)
            signs = np.where(np.random.rand(dim) > 0.5, 1.0, -1.0)
            
            X[j] = p_local + signs * quantum_step
            # Clamp to search bounds
            X[j] = np.clip(X[j], bounds_low, bounds_high)

            # 4. Fitness Evaluation
            act, c_alloc, spd, f_choice = decode_particle_to_fleet_plan(X[j], fleet_vessels, cargo_demand_tonnes, fuels)
            ev = evaluate_fleet_plan(
                fleet_vessels, act, c_alloc, spd, f_choice, distance_nm, weather_condition,
                cargo_demand_tonnes, schedule_deadline_hours, fuels,
                charter_rate_per_hour, carbon_tax_rate, include_carbon_tax, require_shore_power,
                weight_fuel=weight_fuel, weight_cost=weight_cost, weight_ghg=weight_ghg
            )

            if ev["is_feasible"]:
                all_evaluated_solutions.append(ev)

            # Update Personal Best
            if ev["fitness"] < pbest_fitness[j]:
                pbest_fitness[j] = ev["fitness"]
                pbest[j] = np.copy(X[j])
                pbest_evals[j] = ev

                # Update Global Best
                if ev["fitness"] < gbest_fitness:
                    gbest_fitness = ev["fitness"]
                    gbest = np.copy(X[j])
                    gbest_eval = ev

        # Record convergence trajectory
        convergence_history.append({
            "iteration": t + 1,
            "best_fitness": round(float(gbest_fitness), 4),
            "best_fuel_tonnes": gbest_eval["total_fuel_tonnes"],
            "best_cost_usd": gbest_eval["total_cost_usd"],
            "best_wtw_ghg": gbest_eval["total_wtw_ghg_co2e"],
            "beta_parameter": round(float(beta), 3),
            "is_feasible": gbest_eval["is_feasible"]
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Extract Pareto Trade-Off Profiles
    pareto_plans = extract_pareto_tradeoffs(all_evaluated_solutions if all_evaluated_solutions else [gbest_eval], gbest_eval)

    return {
        "algorithm": "QPSO",
        "runtime_ms": round(elapsed_ms, 2),
        "iterations_completed": max_iterations,
        "swarm_size": swarm_size,
        "best_fitness": round(float(gbest_fitness), 4),
        "best_solution": gbest_eval,
        "gbest_vector": gbest.tolist(),
        "convergence_history": convergence_history,
        "feasible_solutions_count": len(all_evaluated_solutions),
        "pareto_plans": pareto_plans
    }


def run_classical_pso_optimization(
    fleet_vessels: List[Dict[str, Any]],
    cargo_demand_tonnes: float,
    distance_nm: float,
    weather_condition: str = "Moderate",
    schedule_deadline_hours: float = 120.0,
    port_available_fuels: Optional[List[str]] = None,
    charter_rate_per_hour: float = 350.0,
    carbon_tax_rate: float = 85.0,
    include_carbon_tax: bool = False,
    require_shore_power: bool = False,
    swarm_size: int = 30,
    max_iterations: int = 60,
    weight_fuel: float = 0.35,
    weight_cost: float = 0.35,
    weight_ghg: float = 0.30,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Classical Particle Swarm Optimization (PSO) with inertia weight and velocity clamping.
    Used as an honest, identical baseline for benchmarking against QPSO.
    """
    start_time = time.perf_counter()
    np.random.seed(random_seed)

    fuels = port_available_fuels if port_available_fuels else list(FUEL_DATABASE.keys())
    n = len(fleet_vessels)
    dim = n * 4

    bounds_low = np.zeros(dim)
    bounds_high = np.zeros(dim)
    for i, ship in enumerate(fleet_vessels):
        idx = i * 4
        bounds_low[idx] = 0.0
        bounds_high[idx] = 1.0
        bounds_low[idx+1] = 0.05
        bounds_high[idx+1] = 1.0
        bounds_low[idx+2] = ship.get("min_speed", 10.0)
        bounds_high[idx+2] = ship.get("max_speed", 22.0)
        bounds_low[idx+3] = 0.0
        compat = ship.get("fuel_compatibility", ["HFO", "VLSFO", "MGO"])
        bounds_high[idx+3] = max(1.0, float(len(compat)))

    v_max = 0.20 * (bounds_high - bounds_low)
    v_min = -v_max

    # Initialize positions and velocities
    X = np.random.uniform(bounds_low, bounds_high, size=(swarm_size, dim))
    V = np.random.uniform(v_min, v_max, size=(swarm_size, dim))

    pbest = np.copy(X)
    pbest_fitness = np.full(swarm_size, np.inf)
    pbest_evals = [None] * swarm_size

    gbest = np.copy(X[0])
    gbest_fitness = np.inf
    gbest_eval = None

    convergence_history = []
    feasible_count = 0

    # Initial evaluation
    for j in range(swarm_size):
        act, c_alloc, spd, f_choice = decode_particle_to_fleet_plan(X[j], fleet_vessels, cargo_demand_tonnes, fuels)
        ev = evaluate_fleet_plan(
            fleet_vessels, act, c_alloc, spd, f_choice, distance_nm, weather_condition,
            cargo_demand_tonnes, schedule_deadline_hours, fuels,
            charter_rate_per_hour, carbon_tax_rate, include_carbon_tax, require_shore_power,
            weight_fuel=weight_fuel, weight_cost=weight_cost, weight_ghg=weight_ghg
        )
        pbest_fitness[j] = ev["fitness"]
        pbest_evals[j] = ev

        if ev["is_feasible"]:
            feasible_count += 1

        if ev["fitness"] < gbest_fitness:
            gbest_fitness = ev["fitness"]
            gbest = np.copy(X[j])
            gbest_eval = ev

    if gbest_eval is None:
        gbest_eval = pbest_evals[0]

    # Classical PSO Loop
    w_max = 0.9
    w_min = 0.4
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter

    for t in range(max_iterations):
        w = w_max - (w_max - w_min) * (t / max_iterations)

        for j in range(swarm_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # Velocity update
            V[j] = w * V[j] + c1 * r1 * (pbest[j] - X[j]) + c2 * r2 * (gbest - X[j])
            V[j] = np.clip(V[j], v_min, v_max)

            # Position update
            X[j] = X[j] + V[j]
            X[j] = np.clip(X[j], bounds_low, bounds_high)

            # Fitness evaluation
            act, c_alloc, spd, f_choice = decode_particle_to_fleet_plan(X[j], fleet_vessels, cargo_demand_tonnes, fuels)
            ev = evaluate_fleet_plan(
                fleet_vessels, act, c_alloc, spd, f_choice, distance_nm, weather_condition,
                cargo_demand_tonnes, schedule_deadline_hours, fuels,
                charter_rate_per_hour, carbon_tax_rate, include_carbon_tax, require_shore_power,
                weight_fuel=weight_fuel, weight_cost=weight_cost, weight_ghg=weight_ghg
            )

            if ev["is_feasible"]:
                feasible_count += 1

            if ev["fitness"] < pbest_fitness[j]:
                pbest_fitness[j] = ev["fitness"]
                pbest[j] = np.copy(X[j])
                pbest_evals[j] = ev

                if ev["fitness"] < gbest_fitness:
                    gbest_fitness = ev["fitness"]
                    gbest = np.copy(X[j])
                    gbest_eval = ev

        convergence_history.append({
            "iteration": t + 1,
            "best_fitness": round(float(gbest_fitness), 4),
            "best_fuel_tonnes": gbest_eval["total_fuel_tonnes"],
            "best_cost_usd": gbest_eval["total_cost_usd"],
            "best_wtw_ghg": gbest_eval["total_wtw_ghg_co2e"],
            "inertia_weight": round(float(w), 3),
            "is_feasible": gbest_eval["is_feasible"]
        })

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "algorithm": "Classical PSO",
        "runtime_ms": round(elapsed_ms, 2),
        "iterations_completed": max_iterations,
        "swarm_size": swarm_size,
        "best_fitness": round(float(gbest_fitness), 4),
        "best_solution": gbest_eval,
        "convergence_history": convergence_history,
        "feasible_solutions_count": feasible_count
    }


def extract_pareto_tradeoffs(
    candidate_solutions: List[Dict[str, Any]],
    fallback_solution: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts structured operational trade-off plans from the candidate pool:
      - Plan A: Eco Green Optimum (Lowest Lifecycle GHG & Fuel)
      - Plan B: Balanced Economic Optimum (Best compromise between Cost, GHG, and Schedule)
      - Plan C: Fast Transit Commercial (Lowest Transit Time meeting deadline)
    """
    if not candidate_solutions:
        return {
            "Plan A": {"name": "Plan A — Green / Decarbonization", "solution": fallback_solution, "tag": "ECO GREEN"},
            "Plan B": {"name": "Plan B — Balanced Fleet Optimum", "solution": fallback_solution, "tag": "BALANCED"},
            "Plan C": {"name": "Plan C — Fast Express Schedule", "solution": fallback_solution, "tag": "FAST SCHEDULE"}
        }

    # Filter feasible solutions
    feasible_pool = [s for s in candidate_solutions if s.get("is_feasible", True)]
    if not feasible_pool:
        feasible_pool = candidate_solutions

    # Plan A: Minimum Lifecycle GHG
    plan_a_sol = min(feasible_pool, key=lambda s: s["total_wtw_ghg_co2e"])
    # Plan B: Minimum Total Cost / Composite Objective
    plan_b_sol = min(feasible_pool, key=lambda s: s["total_cost_usd"])
    # Plan C: Minimum Transit Time
    plan_c_sol = min(feasible_pool, key=lambda s: s["max_transit_time_hrs"])

    return {
        "Plan A": {
            "name": "Plan A — Green / Decarbonization Focus",
            "tag": "MINIMUM EMISSIONS",
            "description": "Slow steaming and renewable fuel prioritization minimizing lifecycle Well-to-Wake GHG.",
            "solution": plan_a_sol
        },
        "Plan B": {
            "name": "Plan B — Balanced Fleet Optimum",
            "tag": "COST-EFFECTIVE OPTIMUM",
            "description": "Optimal operational equilibrium balancing bunker expenditure, time charter, and CII grade.",
            "solution": plan_b_sol
        },
        "Plan C": {
            "name": "Plan C — Fast Express Schedule",
            "tag": "MINIMUM TRANSIT TIME",
            "description": "Higher cruise speed achieving earliest port arrival within designated delivery window.",
            "solution": plan_c_sol
        }
    }


def run_full_benchmark(
    fleet_vessels: List[Dict[str, Any]],
    cargo_demand_tonnes: float,
    distance_nm: float,
    weather_condition: str = "Moderate",
    schedule_deadline_hours: float = 120.0,
    port_available_fuels: Optional[List[str]] = None,
    charter_rate_per_hour: float = 350.0,
    carbon_tax_rate: float = 85.0,
    include_carbon_tax: bool = False,
    require_shore_power: bool = False,
    swarm_size: int = 30,
    max_iterations: int = 50,
    weight_fuel: float = 0.35,
    weight_cost: float = 0.35,
    weight_ghg: float = 0.30,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes a side-by-side benchmark between Classical PSO and Quantum-Inspired QPSO
    using identical input parameters, constraints, and random seeds.
    """
    # 1. Run Classical PSO
    pso_res = run_classical_pso_optimization(
        fleet_vessels=fleet_vessels,
        cargo_demand_tonnes=cargo_demand_tonnes,
        distance_nm=distance_nm,
        weather_condition=weather_condition,
        schedule_deadline_hours=schedule_deadline_hours,
        port_available_fuels=port_available_fuels,
        charter_rate_per_hour=charter_rate_per_hour,
        carbon_tax_rate=carbon_tax_rate,
        include_carbon_tax=include_carbon_tax,
        require_shore_power=require_shore_power,
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        weight_fuel=weight_fuel,
        weight_cost=weight_cost,
        weight_ghg=weight_ghg,
        random_seed=random_seed
    )

    # 2. Run Quantum-Inspired QPSO
    qpso_res = run_qpso_optimization(
        fleet_vessels=fleet_vessels,
        cargo_demand_tonnes=cargo_demand_tonnes,
        distance_nm=distance_nm,
        weather_condition=weather_condition,
        schedule_deadline_hours=schedule_deadline_hours,
        port_available_fuels=port_available_fuels,
        charter_rate_per_hour=charter_rate_per_hour,
        carbon_tax_rate=carbon_tax_rate,
        include_carbon_tax=include_carbon_tax,
        require_shore_power=require_shore_power,
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        weight_fuel=weight_fuel,
        weight_cost=weight_cost,
        weight_ghg=weight_ghg,
        random_seed=random_seed
    )

    # 3. Calculate genuine comparison metrics
    pso_sol = pso_res["best_solution"]
    qpso_sol = qpso_res["best_solution"]

    fuel_delta = qpso_sol["total_fuel_tonnes"] - pso_sol["total_fuel_tonnes"]
    cost_delta = qpso_sol["total_cost_usd"] - pso_sol["total_cost_usd"]
    ghg_delta = qpso_sol["total_wtw_ghg_co2e"] - pso_sol["total_wtw_ghg_co2e"]

    # Convergence speed (iterations to reach within 2% of final fitness)
    def find_convergence_step(history):
        final_fit = history[-1]["best_fitness"]
        threshold = final_fit * 1.02
        for item in history:
            if item["best_fitness"] <= threshold:
                return item["iteration"]
        return len(history)

    pso_conv_iter = find_convergence_step(pso_res["convergence_history"])
    qpso_conv_iter = find_convergence_step(qpso_res["convergence_history"])

    comparison_table = pd.DataFrame([
        {
            "Metric": "Best Objective Fitness",
            "Classical PSO": f"{pso_res['best_fitness']:.4f}",
            "Quantum-Inspired QPSO": f"{qpso_res['best_fitness']:.4f}",
            "Advantage / Delta": f"{(qpso_res['best_fitness'] - pso_res['best_fitness']):+.4f}"
        },
        {
            "Metric": "Total Fleet Fuel (tonnes)",
            "Classical PSO": f"{pso_sol['total_fuel_tonnes']:,.1f} t",
            "Quantum-Inspired QPSO": f"{qpso_sol['total_fuel_tonnes']:,.1f} t",
            "Advantage / Delta": f"{fuel_delta:+,.1f} t ({(fuel_delta / max(1, pso_sol['total_fuel_tonnes']) * 100.0):+.1f}%)"
        },
        {
            "Metric": "Total Operational Cost ($)",
            "Classical PSO": f"${pso_sol['total_cost_usd']:,.0f}",
            "Quantum-Inspired QPSO": f"${qpso_sol['total_cost_usd']:,.0f}",
            "Advantage / Delta": f"${cost_delta:+,.0f} ({(cost_delta / max(1, pso_sol['total_cost_usd']) * 100.0):+.1f}%)"
        },
        {
            "Metric": "Lifecycle Well-to-Wake GHG",
            "Classical PSO": f"{pso_sol['total_wtw_ghg_co2e']:,.1f} t CO2e",
            "Quantum-Inspired QPSO": f"{qpso_sol['total_wtw_ghg_co2e']:,.1f} t CO2e",
            "Advantage / Delta": f"{ghg_delta:+,.1f} t CO2e ({(ghg_delta / max(1, pso_sol['total_wtw_ghg_co2e']) * 100.0):+.1f}%)"
        },
        {
            "Metric": "Algorithm Runtime",
            "Classical PSO": f"{pso_res['runtime_ms']:.1f} ms",
            "Quantum-Inspired QPSO": f"{qpso_res['runtime_ms']:.1f} ms",
            "Advantage / Delta": f"{(qpso_res['runtime_ms'] - pso_res['runtime_ms']):+.1f} ms"
        },
        {
            "Metric": "Iterations to 98% Convergence",
            "Classical PSO": f"{pso_conv_iter} iterations",
            "Quantum-Inspired QPSO": f"{qpso_conv_iter} iterations",
            "Advantage / Delta": f"{(qpso_conv_iter - pso_conv_iter):+d} iterations"
        },
        {
            "Metric": "Feasible States Explored",
            "Classical PSO": f"{pso_res['feasible_solutions_count']} samples",
            "Quantum-Inspired QPSO": f"{qpso_res['feasible_solutions_count']} samples",
            "Advantage / Delta": f"{(qpso_res['feasible_solutions_count'] - pso_res['feasible_solutions_count']):+d}"
        }
    ])

    return {
        "pso_result": pso_res,
        "qpso_result": qpso_res,
        "comparison_table": comparison_table,
        "pso_conv_iter": pso_conv_iter,
        "qpso_conv_iter": qpso_conv_iter
    }
