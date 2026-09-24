"""
========================================================================================
EGREEN QUANTA — EMISSIONS, IMO COMPLIANCE & FINANCIAL COSTING MODULE
========================================================================================
Implements IMO GHG Strategy Emission Factors, Well-to-Wake Lifecycle GHG accounting,
Carbon Intensity Indicator (CII) rating, and EU ETS Carbon Tax calculations.
"""

from typing import Dict, Any, Optional
from utils.fuel_scenarios import calculate_lifecycle_emissions, get_cii_grade, FUEL_DATABASE

DEFAULT_BUNKER_PRICE_USD = 620.0  # VLSFO benchmark
CARBON_TAX_PER_TONNE_USD = 85.0   # EU ETS benchmark


def calculate_emissions_and_costs(
    fuel_tonnes: float,
    bunker_price_usd: float = DEFAULT_BUNKER_PRICE_USD,
    emission_factor: float = 3.17,
    cargo_tonnes: float = 50000.0,
    distance_nm: float = 1000.0,
    include_carbon_tax: bool = False,
    carbon_tax_rate: float = CARBON_TAX_PER_TONNE_USD,
    fuel_type: str = "VLSFO",
    berth_hours: float = 0.0,
    use_shore_power: bool = False
) -> Dict[str, Any]:
    """
    Computes direct combustion CO2 (TtW), upstream WtT, total lifecycle GHG (WtW CO2e),
    direct bunker fuel cost, carbon tax liabilities, and IMO CII rating.
    """
    # Use the centralized lifecycle calculator
    lifecycle_res = calculate_lifecycle_emissions(
        fuel_type=fuel_type,
        fuel_tonnes=fuel_tonnes,
        cargo_tonnes=cargo_tonnes,
        distance_nm=distance_nm,
        bunker_price_usd=bunker_price_usd,
        include_carbon_tax=include_carbon_tax,
        carbon_tax_rate=carbon_tax_rate,
        berth_hours=berth_hours,
        use_shore_power=use_shore_power
    )

    # Return standardized dictionary containing both legacy and lifecycle fields
    return {
        "fuel_tonnes": lifecycle_res["fuel_tonnes"],
        "fuel_type": fuel_type,
        "fuel_name": lifecycle_res["fuel_name"],
        "bunker_price_usd": lifecycle_res["unit_price_usd_tonne"],
        "co2_tonnes": lifecycle_res["ttw_co2_tonnes"],
        "ttw_ghg_co2e": lifecycle_res["ttw_ghg_co2e_tonnes"],
        "wtt_ghg_co2e": lifecycle_res["wtt_ghg_co2e_tonnes"],
        "wtw_lifecycle_co2e": lifecycle_res["wtw_lifecycle_ghg_co2e_tonnes"],
        "direct_fuel_cost_usd": lifecycle_res["direct_bunker_cost_usd"],
        "carbon_tax_cost_usd": lifecycle_res["carbon_tax_cost_usd"],
        "total_cost_usd": lifecycle_res["total_fuel_cost_usd"],
        "cii_value_g_t_nm": lifecycle_res["cii_value_g_co2e_t_nm"],
        "cii_rating": lifecycle_res["cii_rating"],
        "cii_color": lifecycle_res["cii_color"],
        "cii_badge": lifecycle_res["cii_badge"],
        "use_shore_power": use_shore_power,
        "port_co2e_saved_tonnes": lifecycle_res["port_co2e_saved_tonnes"],
        "emission_factor": emission_factor
    }


def calculate_emission_cost_comparison(
    normal_fuel_tonnes: float,
    slow_fuel_tonnes: float,
    bunker_price_usd: float = DEFAULT_BUNKER_PRICE_USD,
    cargo_tonnes: float = 50000.0,
    distance_nm: float = 1000.0,
    include_carbon_tax: bool = False,
    fuel_type: str = "VLSFO"
) -> Dict[str, Any]:
    """Generates side-by-side comparison between Normal Speed and Slow Steaming."""
    normal_metrics = calculate_emissions_and_costs(
        normal_fuel_tonnes, bunker_price_usd, 3.17, cargo_tonnes, distance_nm, include_carbon_tax, fuel_type=fuel_type
    )
    slow_metrics = calculate_emissions_and_costs(
        slow_fuel_tonnes, bunker_price_usd, 3.17, cargo_tonnes, distance_nm, include_carbon_tax, fuel_type=fuel_type
    )

    fuel_saved = max(0.0, normal_metrics["fuel_tonnes"] - slow_metrics["fuel_tonnes"])
    co2_reduced = max(0.0, normal_metrics["wtw_lifecycle_co2e"] - slow_metrics["wtw_lifecycle_co2e"])
    cost_saved = max(0.0, normal_metrics["total_cost_usd"] - slow_metrics["total_cost_usd"])
    
    pct_fuel_saved = (fuel_saved / normal_metrics["fuel_tonnes"] * 100.0) if normal_metrics["fuel_tonnes"] > 0 else 0.0
    pct_co2_reduced = (co2_reduced / normal_metrics["wtw_lifecycle_co2e"] * 100.0) if normal_metrics["wtw_lifecycle_co2e"] > 0 else 0.0
    pct_cost_saved = (cost_saved / normal_metrics["total_cost_usd"] * 100.0) if normal_metrics["total_cost_usd"] > 0 else 0.0

    return {
        "normal": normal_metrics,
        "slow": slow_metrics,
        "fuel_saved_tonnes": round(fuel_saved, 2),
        "co2_reduced_tonnes": round(co2_reduced, 2),
        "cost_saved_usd": round(cost_saved, 2),
        "pct_fuel_saved": round(pct_fuel_saved, 1),
        "pct_co2_reduced": round(pct_co2_reduced, 1),
        "pct_cost_saved": round(pct_cost_saved, 1)
    }
