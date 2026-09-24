"""
========================================================================================
EGREEN QUANTA — MARITIME FLEET OPERATIONAL CONSTRAINT ENGINE
========================================================================================
Validates and enforces real-world commercial, hydrodynamic, and regulatory constraints:
  1. Cargo Demand Satisfaction: Sum of allocated cargo >= Total required cargo
  2. Vessel Deadweight Capacity: Cargo per vessel <= Max DWT payload
  3. Operating Speed Envelope: Min Speed <= Selected Speed <= Max Speed
  4. Schedule Delivery Window: Max Transit Time <= Voyage Delivery Deadline
  5. Vessel Fleet Availability: Offline/maintenance vessels cannot be deployed
  6. Engine Fuel Compatibility: Vessel can only combust supported fuel types
  7. Port Bunkering Availability: Fuel must be available at port/terminal
  8. Shore Power Compatibility: Vessel cold-ironing equipment check
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class FleetConstraintEngine:
    """
    Evaluates multi-vessel operational feasibility and computes mathematically
    rigorous penalty scores for constrained optimization algorithms.
    """

    def __init__(
        self,
        penalty_cargo_deficit: float = 50000.0,     # High penalty per tonne unmet demand
        penalty_capacity_excess: float = 25000.0,    # Penalty per tonne overload
        penalty_schedule_delay: float = 1500.0,     # Penalty per hour delayed beyond deadline
        penalty_incompatible_fuel: float = 100000.0, # Hard penalty for invalid fuel
        penalty_unavailable_ship: float = 100000.0   # Hard penalty for deploying offline ship
    ):
        self.p_cargo = penalty_cargo_deficit
        self.p_cap = penalty_capacity_excess
        self.p_delay = penalty_schedule_delay
        self.p_fuel = penalty_incompatible_fuel
        self.p_unavail = penalty_unavailable_ship

    def evaluate_plan(
        self,
        fleet_vessels: List[Dict[str, Any]],
        cargo_allocations: List[float],
        speeds: List[float],
        fuel_choices: List[str],
        active_flags: List[bool],
        voyage_distance_nm: float,
        cargo_demand_tonnes: float,
        schedule_deadline_hours: float,
        port_available_fuels: List[str],
        require_shore_power: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates full fleet operational state against all real-world maritime constraints.
        Returns feasibility boolean, list of explicit violations, and penalty cost.
        """
        violations: List[str] = []
        penalties: float = 0.0

        total_allocated_cargo = 0.0
        max_transit_time_hrs = 0.0
        num_active = 0

        for i, ship in enumerate(fleet_vessels):
            is_active = active_flags[i]
            cargo = max(0.0, float(cargo_allocations[i])) if is_active else 0.0
            speed = float(speeds[i])
            fuel = fuel_choices[i]
            
            ship_name = ship.get("name", f"Vessel-{i+1}")
            max_dwt = float(ship.get("max_cargo", ship.get("dwt", 80000.0)))
            min_speed = float(ship.get("min_speed", 10.0))
            max_speed = float(ship.get("max_speed", 22.0))
            is_available = ship.get("status", "Active") == "Active" and ship.get("available", True)
            compat_fuels = ship.get("fuel_compatibility", ["HFO", "VLSFO", "MGO"])

            if is_active:
                num_active += 1
                total_allocated_cargo += cargo

                # 1. Vessel Availability Constraint
                if not is_available:
                    violations.append(f"{ship_name} is marked UNAVAILABLE/MAINTENANCE but assigned cargo.")
                    penalties += self.p_unavail

                # 2. Vessel Capacity Constraint
                if cargo > max_dwt * 1.001:
                    excess = cargo - max_dwt
                    violations.append(f"{ship_name} overloaded by {excess:,.0f} tonnes (Cap: {max_dwt:,.0f} t, Alloc: {cargo:,.0f} t).")
                    penalties += excess * self.p_cap

                # 3. Speed Boundary Limits
                if speed < (min_speed - 0.01) or speed > (max_speed + 0.01):
                    violations.append(f"{ship_name} speed {speed:.1f} kn violates operating bounds [{min_speed:.1f} - {max_speed:.1f} kn].")
                    speed_dev = max(0.0, min_speed - speed) + max(0.0, speed - max_speed)
                    penalties += speed_dev * 10000.0

                # 4. Engine Fuel Compatibility Constraint
                if fuel not in compat_fuels:
                    violations.append(f"{ship_name} engine incompatible with {fuel} (Supported: {', '.join(compat_fuels)}).")
                    penalties += self.p_fuel

                # 5. Port Bunkering Availability Constraint
                if fuel not in port_available_fuels:
                    violations.append(f"Fuel {fuel} chosen for {ship_name} is NOT available at bunkering port.")
                    penalties += self.p_fuel * 0.5

                # 6. Shore Power Compatibility Check
                if require_shore_power and not ship.get("has_shore_power", True):
                    violations.append(f"{ship_name} lacks shore power cold-ironing connection required at berth.")
                    penalties += 20000.0

                # Transit Time Calculation
                transit_hrs = voyage_distance_nm / max(1.0, speed)
                if transit_hrs > max_transit_time_hrs:
                    max_transit_time_hrs = transit_hrs

        # 7. Total Fleet Cargo Demand Satisfaction
        cargo_deficit = cargo_demand_tonnes - total_allocated_cargo
        if cargo_deficit > 1.0:  # Allow 1 tonne tolerance for numerical floating point
            violations.append(f"Fleet unmet cargo demand deficit: {cargo_deficit:,.0f} tonnes (Demand: {cargo_demand_tonnes:,.0f} t, Allocated: {total_allocated_cargo:,.0f} t).")
            penalties += cargo_deficit * self.p_cargo

        # 8. Schedule Delivery Window Deadline Constraint
        schedule_delay_hrs = max(0.0, max_transit_time_hrs - schedule_deadline_hours)
        if schedule_delay_hrs > 0.1 and schedule_deadline_hours > 0:
            violations.append(f"Fleet ETA of {max_transit_time_hrs:.1f} hrs exceeds delivery deadline of {schedule_deadline_hours:.1f} hrs (Delay: {schedule_delay_hrs:.1f} hrs).")
            penalties += schedule_delay_hrs * self.p_delay

        # Zero active ships with non-zero demand is invalid
        if num_active == 0 and cargo_demand_tonnes > 0:
            violations.append("No active vessels assigned to fulfill voyage cargo demand.")
            penalties += self.p_cargo * cargo_demand_tonnes

        is_feasible = (len(violations) == 0)

        return {
            "is_feasible": is_feasible,
            "violations_count": len(violations),
            "violations": violations,
            "penalty_score": round(penalties, 2),
            "total_allocated_cargo_tonnes": round(total_allocated_cargo, 1),
            "cargo_deficit_tonnes": round(max(0.0, cargo_deficit), 1),
            "max_transit_time_hrs": round(max_transit_time_hrs, 1),
            "schedule_delay_hrs": round(schedule_delay_hrs, 1),
            "num_active_vessels": num_active
        }


# Global default instance
default_constraint_engine = FleetConstraintEngine()
