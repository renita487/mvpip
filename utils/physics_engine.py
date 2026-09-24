"""
========================================================================================
EGREEN QUANTA — NAVAL HYDRODYNAMICS & MARINE PHYSICS ENGINE
========================================================================================
Physics-based fuel consumption modeling based on:
  - Admiralty Coefficient Formulation: P = (Delta^(2/3) * V^3) / C_adm
  - Displacement (Delta) based on lightweight + payload cargo load
  - Specific Fuel Oil Consumption (SFOC) & Engine Maximum Continuous Rating (MCR)
  - Weather & Sea State Added Resistance Penalties (Beaufort Sea States 0 to 7+)
  - Alternative Fuel Energy Equivalence via Lower Heating Value (LHV)
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from utils.fuel_scenarios import get_fuel_spec, calculate_fuel_energy_mass

# Standard baseline vessel engineering specifications
VESSEL_SPECS: Dict[str, Dict[str, Any]] = {
    "Container Ship": {
        "lightweight": 25000.0,       # Tonnes (empty ship weight)
        "max_cargo": 85000.0,         # Tonnes (max DWT payload)
        "default_cargo": 55000.0,     # Tonnes
        "design_speed": 21.0,         # Knots
        "min_speed": 10.0,            # Knots
        "max_speed": 22.5,            # Knots
        "design_power_kw": 48000.0,   # kW (MCR)
        "c_admiralty": 550.0,         # Admiralty Coefficient
        "sfoc_g_kwh": 178.0,          # Base SFOC on VLSFO (g/kWh)
        "auxiliary_fuel_t_day": 3.5,  # Generator auxiliary fuel consumption (t/day)
        "auxiliary_power_kw": 1800.0, # Auxiliary electrical demand (kW)
        "fuel_compatibility": ["HFO", "VLSFO", "MGO", "LNG", "Methanol"],
        "has_shore_power": True,
        "icon": "🚢"
    },
    "Bulk Carrier": {
        "lightweight": 18000.0,
        "max_cargo": 120000.0,
        "default_cargo": 85000.0,
        "design_speed": 14.5,
        "min_speed": 9.0,
        "max_speed": 16.5,
        "design_power_kw": 16500.0,
        "c_admiralty": 620.0,
        "sfoc_g_kwh": 172.0,
        "auxiliary_fuel_t_day": 2.0,
        "auxiliary_power_kw": 950.0,
        "fuel_compatibility": ["HFO", "VLSFO", "MGO", "Ammonia"],
        "has_shore_power": True,
        "icon": "⛴️"
    },
    "Tanker": {
        "lightweight": 28000.0,
        "max_cargo": 160000.0,
        "default_cargo": 110000.0,
        "design_speed": 15.0,
        "min_speed": 9.5,
        "max_speed": 17.0,
        "design_power_kw": 22000.0,
        "c_admiralty": 600.0,
        "sfoc_g_kwh": 175.0,
        "auxiliary_fuel_t_day": 2.8,
        "auxiliary_power_kw": 1400.0,
        "fuel_compatibility": ["HFO", "VLSFO", "MGO", "LNG"],
        "has_shore_power": False,
        "icon": "🛢️"
    },
    "General Cargo": {
        "lightweight": 8000.0,
        "max_cargo": 35000.0,
        "default_cargo": 20000.0,
        "design_speed": 13.5,
        "min_speed": 8.5,
        "max_speed": 15.5,
        "design_power_kw": 9500.0,
        "c_admiralty": 520.0,
        "sfoc_g_kwh": 182.0,
        "auxiliary_fuel_t_day": 1.5,
        "auxiliary_power_kw": 600.0,
        "fuel_compatibility": ["VLSFO", "MGO", "Methanol", "Hydrogen"],
        "has_shore_power": True,
        "icon": "📦"
    }
}

WEATHER_FACTORS: Dict[str, Dict[str, Any]] = {
    "Calm": {
        "multiplier": 1.00,
        "beaufort_scale": "0 - 3 (Light breeze, wave < 0.8m, wind < 10 kn)",
        "added_resistance_pct": "0%",
        "wind_knots": 8.0,
        "wave_m": 0.8,
        "color": "#10B981"
    },
    "Moderate": {
        "multiplier": 1.15,
        "beaufort_scale": "4 - 6 (Moderate breeze to gale, wave 1.5 - 3.0m, wind 15-25 kn)",
        "added_resistance_pct": "+15%",
        "wind_knots": 18.0,
        "wave_m": 2.2,
        "color": "#F59E0B"
    },
    "Rough": {
        "multiplier": 1.35,
        "beaufort_scale": "7+ (High waves > 4.0m, strong swell & headwind > 30 kn)",
        "added_resistance_pct": "+35%",
        "wind_knots": 32.0,
        "wave_m": 4.5,
        "color": "#EF4444"
    }
}


def calculate_hydrodynamic_fuel(
    vessel_type: str,
    speed_knots: float,
    cargo_tonnes: float,
    weather_condition: str,
    distance_nm: float = 1000.0,
    fuel_type: str = "VLSFO",
    custom_specs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes exact hydrodynamic propulsion power, fuel consumption (tonnes),
    transit duration, and auxiliary loads using Admiralty naval architecture mechanics.
    """
    specs = custom_specs if custom_specs else VESSEL_SPECS.get(vessel_type, VESSEL_SPECS["Container Ship"])
    weather = WEATHER_FACTORS.get(weather_condition, WEATHER_FACTORS["Moderate"])

    # 1. Total Displacement (delta in tonnes)
    max_c = specs.get("max_cargo", specs.get("dwt", 80000.0))
    lightweight = specs.get("lightweight", 20000.0)
    displacement = lightweight + max(0.0, min(cargo_tonnes, max_c))

    # 2. Admiralty Power Estimation: P = (Delta^(2/3) * V^3) / C_adm
    c_adm = specs.get("c_admiralty", 550.0)
    design_power = specs.get("design_power_kw", 30000.0)
    
    speed_safe = max(1.0, speed_knots)
    propulsion_power_kw = ((displacement ** (2/3)) * (speed_safe ** 3)) / c_adm
    propulsion_power_kw = min(propulsion_power_kw, design_power * 1.10)

    # 3. Weather resistance multiplier
    weather_mult = weather["multiplier"]
    effective_power_kw = propulsion_power_kw * weather_mult

    # 4. Specific Fuel Oil Consumption & Base VLSFO Fuel Mass
    base_sfoc = specs.get("sfoc_g_kwh", 175.0)
    base_fuel_day = (effective_power_kw * base_sfoc * 24.0) / 1_000_000.0
    aux_fuel_day = specs.get("auxiliary_fuel_t_day", 2.5)

    # 5. Voyage transit duration
    voyage_hours = distance_nm / speed_safe
    voyage_days = voyage_hours / 24.0

    # Total baseline (VLSFO) mass required
    base_total_fuel = (base_fuel_day + aux_fuel_day) * voyage_days

    # 6. Alternative Fuel Energy Adjustment
    # Convert baseline VLSFO mass into required mass of target fuel_type
    target_total_fuel = calculate_fuel_energy_mass(base_total_fuel, "VLSFO", fuel_type)
    target_fuel_day = (target_total_fuel / max(0.01, voyage_days))

    fuel_per_nm = target_total_fuel / max(1.0, distance_nm)

    return {
        "vessel_type": vessel_type,
        "fuel_type": fuel_type,
        "speed_knots": float(round(speed_knots, 2)),
        "cargo_tonnes": float(round(cargo_tonnes, 1)),
        "weather_condition": weather_condition,
        "weather_multiplier": weather_mult,
        "displacement_tonnes": float(round(displacement, 1)),
        "effective_power_kw": float(round(effective_power_kw, 1)),
        "fuel_per_day_tonnes": float(round(target_fuel_day, 2)),
        "aux_fuel_day_tonnes": float(round(aux_fuel_day, 2)),
        "voyage_distance_nm": float(distance_nm),
        "voyage_hours": float(round(voyage_hours, 1)),
        "voyage_days": float(round(voyage_days, 2)),
        "total_voyage_fuel_tonnes": float(round(target_total_fuel, 2)),
        "fuel_per_nm_tonnes": float(round(fuel_per_nm, 4)),
        "auxiliary_power_kw": float(specs.get("auxiliary_power_kw", 1200.0)),
        "has_shore_power": specs.get("has_shore_power", True)
    }


def compute_slow_steaming_comparison(
    vessel_type: str,
    speed_knots: float,
    cargo_tonnes: float,
    weather_condition: str,
    distance_nm: float = 1000.0,
    speed_delta: float = 2.0,
    fuel_type: str = "VLSFO"
) -> Dict[str, Any]:
    """Compares baseline speed against Slow Steaming (-delta knots) over the same voyage."""
    normal_res = calculate_hydrodynamic_fuel(
        vessel_type, speed_knots, cargo_tonnes, weather_condition, distance_nm, fuel_type
    )
    slow_speed = max(8.0, speed_knots - speed_delta)
    slow_res = calculate_hydrodynamic_fuel(
        vessel_type, slow_speed, cargo_tonnes, weather_condition, distance_nm, fuel_type
    )

    fuel_saved = max(0.0, normal_res["total_voyage_fuel_tonnes"] - slow_res["total_voyage_fuel_tonnes"])
    fuel_saved_pct = (fuel_saved / normal_res["total_voyage_fuel_tonnes"]) * 100.0 if normal_res["total_voyage_fuel_tonnes"] > 0 else 0.0
    extra_hours = max(0.0, slow_res["voyage_hours"] - normal_res["voyage_hours"])

    return {
        "normal": normal_res,
        "slow": slow_res,
        "slow_speed_knots": slow_speed,
        "fuel_saved_voyage_tonnes": round(fuel_saved, 2),
        "fuel_saved_pct": round(fuel_saved_pct, 1),
        "fuel_saved_daily_tonnes": round(max(0.0, normal_res["fuel_per_day_tonnes"] - slow_res["fuel_per_day_tonnes"]), 2),
        "extra_voyage_hours": round(extra_hours, 1)
    }


def generate_speed_curve_data(
    vessel_type: str,
    cargo_tonnes: float,
    weather_condition: str,
    distance_nm: float = 1000.0,
    min_speed: float = 10.0,
    max_speed: float = 22.0,
    step: float = 0.5,
    fuel_type: str = "VLSFO"
) -> Dict[str, list]:
    """Generates data points for non-linear speed-fuel sensitivity curves."""
    speeds = np.arange(min_speed, max_speed + step, step)
    daily_fuels = []
    voyage_fuels = []
    powers_kw = []

    for s in speeds:
        res = calculate_hydrodynamic_fuel(vessel_type, float(s), cargo_tonnes, weather_condition, distance_nm, fuel_type)
        daily_fuels.append(res["fuel_per_day_tonnes"])
        voyage_fuels.append(res["total_voyage_fuel_tonnes"])
        powers_kw.append(res["effective_power_kw"])

    return {
        "speeds": [float(round(s, 1)) for s in speeds],
        "daily_fuels": [float(round(f, 2)) for f in daily_fuels],
        "voyage_fuels": [float(round(f, 2)) for f in voyage_fuels],
        "powers_kw": [float(round(p, 1)) for p in powers_kw]
    }
