"""
========================================================================================
EGREEN QUANTA — CENTRALIZED ALTERNATIVE-FUEL SCENARIO & LIFECYCLE GHG ENGINE
ISO 14040 / IMO MEPC.308(73) / IMO 2023 GHG Strategy Standards
========================================================================================
Provides authoritative fuel characteristics, Lower Heating Values (LHV), benchmark prices,
and full lifecycle emissions accounting:
  - Tank-to-Wake (TtW): Direct operational combustion emissions (CO2, CH4 slip, N2O)
  - Well-to-Tank (WtT): Upstream extraction, processing, refining, and supply chain GHG
  - Well-to-Wake (WtW): Full cradle-to-grave lifecycle GHG in CO2-equivalent (CO2e)
  - Shore Power (Cold Ironing): Port berth auxiliary electrification
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Global Warming Potentials (IPCC AR5/AR6 100-year GWP)
GWP_CO2 = 1.0
GWP_CH4 = 28.0   # Methane slip
GWP_N2O = 265.0  # Nitrous oxide

# Centralized Maritime Fuel Characteristics & Life Cycle Assessment (LCA) Database
FUEL_DATABASE: Dict[str, Dict[str, Any]] = {
    "HFO": {
        "name": "Heavy Fuel Oil (HFO)",
        "category": "Conventional Fossil",
        "lhv_mj_kg": 40.2,
        "default_price_usd_tonne": 580.0,
        "ttw_co2_factor": 3.114,       # t CO2 / t fuel
        "ttw_ch4_factor": 0.00005,     # t CH4 / t fuel
        "ttw_n2o_factor": 0.00015,     # t N2O / t fuel
        "wtt_ghg_factor": 0.580,       # t CO2e / t fuel (Upstream extraction & refining)
        "bunkering_availability": "Global (High)",
        "storage_temp": "Ambient (Heated)",
        "energy_density_vol_mj_l": 38.6,
        "regulatory_status": "Subject to 0.50% Sulphur Cap (Scrubber required)",
        "description": "Standard high-viscosity residual marine bunker fuel with scrubbers."
    },
    "VLSFO": {
        "name": "Very Low Sulphur Fuel Oil (VLSFO)",
        "category": "Compliant Fossil",
        "lhv_mj_kg": 41.2,
        "default_price_usd_tonne": 620.0,
        "ttw_co2_factor": 3.151,
        "ttw_ch4_factor": 0.00004,
        "ttw_n2o_factor": 0.00014,
        "wtt_ghg_factor": 0.620,
        "bunkering_availability": "Global (High)",
        "storage_temp": "Ambient (Mild heating)",
        "energy_density_vol_mj_l": 37.0,
        "regulatory_status": "IMO 2020 0.50% Sulphur Compliant",
        "description": "Global baseline low-sulphur bunker oil."
    },
    "MGO": {
        "name": "Marine Gas Oil (MGO)",
        "category": "Distillate Fossil",
        "lhv_mj_kg": 42.7,
        "default_price_usd_tonne": 820.0,
        "ttw_co2_factor": 3.206,
        "ttw_ch4_factor": 0.00002,
        "ttw_n2o_factor": 0.00012,
        "wtt_ghg_factor": 0.680,
        "bunkering_availability": "Global (High)",
        "storage_temp": "Ambient",
        "energy_density_vol_mj_l": 36.3,
        "regulatory_status": "ECA (Emission Control Area) Compliant (0.10% S)",
        "description": "Premium distillate marine diesel for coastal and ECA zones."
    },
    "LNG": {
        "name": "Liquefied Natural Gas (LNG)",
        "category": "Transition Fuel",
        "lhv_mj_kg": 49.1,
        "default_price_usd_tonne": 750.0,
        "ttw_co2_factor": 2.750,       # Lower carbon intensity per tonne
        "ttw_ch4_factor": 0.01250,     # Methane slip (~1.25% in 4-stroke / 0.2% in 2-stroke)
        "ttw_n2o_factor": 0.00008,
        "wtt_ghg_factor": 0.850,       # Liquefaction & cryo-transport upstream
        "bunkering_availability": "Major Hubs (Expanding)",
        "storage_temp": "Cryogenic (-162°C)",
        "energy_density_vol_mj_l": 21.6,
        "regulatory_status": "SOx/PM zero, low NOx, GHG reduction ~20-23% TtW",
        "description": "Cryogenic methane reducing operational CO2, with methane slip accounted."
    },
    "Methanol": {
        "name": "Green / Bio-Methanol (CH3OH)",
        "category": "Renewable Alternative",
        "lhv_mj_kg": 19.9,
        "default_price_usd_tonne": 850.0,
        "ttw_co2_factor": 1.375,       # Biogenic / recirculated CO2
        "ttw_ch4_factor": 0.00001,
        "ttw_n2o_factor": 0.00005,
        "wtt_ghg_factor": 0.350,       # Bio-feedstock / E-methanol synthesis upstream
        "bunkering_availability": "Selected Bunkering Hubs (Rotterdam, Singapore, Shanghai)",
        "storage_temp": "Ambient Liquid",
        "energy_density_vol_mj_l": 15.8,
        "regulatory_status": "Zero-SOx, Low NOx, Net-zero lifecycle ready",
        "description": "Liquid renewable fuel compatible with retrofitted dual-fuel engines."
    },
    "Ammonia": {
        "name": "Green Ammonia (NH3)",
        "category": "Zero-Carbon Carrier",
        "lhv_mj_kg": 18.6,
        "default_price_usd_tonne": 980.0,
        "ttw_co2_factor": 0.000,       # Zero carbon in fuel molecule
        "ttw_ch4_factor": 0.00000,
        "ttw_n2o_factor": 0.00045,     # Trace N2O from combustion pilot fuel
        "wtt_ghg_factor": 0.280,       # Renewable Haber-Bosch green electricity upstream
        "bunkering_availability": "Prototype / Early Commercial (2025+)",
        "storage_temp": "Refrigerated (-33°C / Pressurized)",
        "energy_density_vol_mj_l": 12.7,
        "regulatory_status": "Zero-Carbon Tank-to-Wake, safety protocols strictly regulated",
        "description": "Zero-carbon fuel molecule generated from green hydrogen and nitrogen."
    },
    "Hydrogen": {
        "name": "Liquid Green Hydrogen (LH2)",
        "category": "Zero-Carbon Carrier",
        "lhv_mj_kg": 120.0,
        "default_price_usd_tonne": 2600.0,
        "ttw_co2_factor": 0.000,
        "ttw_ch4_factor": 0.00000,
        "ttw_n2o_factor": 0.00000,
        "wtt_ghg_factor": 0.420,       # Electrolysis & liquefaction energy
        "bunkering_availability": "Experimental / Specialized Ports",
        "storage_temp": "Cryogenic (-253°C)",
        "energy_density_vol_mj_l": 8.5,
        "regulatory_status": "True Zero Emission at point of use (Fuel Cell / ICE)",
        "description": "High gravimetric energy carrier suitable for fuel cells and clean combustion."
    }
}

# Shore Power (Cold Ironing / Onshore Power Supply) Defaults
SHORE_POWER_DEFAULTS = {
    "grid_carbon_intensity_g_kwh": 420.0,  # Global grid average (g CO2e / kWh)
    "green_grid_carbon_g_kwh": 65.0,       # Renewable grid (g CO2e / kWh)
    "electricity_tariff_usd_kwh": 0.18,    # Commercial port electricity tariff ($/kWh)
    "marine_aux_diesel_sfoc_g_kwh": 215.0, # Auxiliary engine SFOC on MDO
}


def get_fuel_spec(fuel_key: str) -> Dict[str, Any]:
    """Retrieves fuel specification dict with safe fallback to VLSFO."""
    return FUEL_DATABASE.get(fuel_key, FUEL_DATABASE["VLSFO"])


def calculate_fuel_energy_mass(
    baseline_fuel_tonnes: float,
    baseline_fuel_type: str,
    target_fuel_type: str
) -> float:
    """
    Calculates equivalent mass (tonnes) of target fuel required to provide
    the exact same propulsive energy output as the baseline fuel mass.
    Energy (MJ) = Mass (kg) * LHV (MJ/kg)
    Target Mass = Baseline Energy / Target LHV
    """
    base_spec = get_fuel_spec(baseline_fuel_type)
    target_spec = get_fuel_spec(target_fuel_type)

    # Total MJ required
    propulsion_energy_mj = (baseline_fuel_tonnes * 1000.0) * base_spec["lhv_mj_kg"]
    # Target tonnes required
    target_tonnes = (propulsion_energy_mj / target_spec["lhv_mj_kg"]) / 1000.0
    return float(round(target_tonnes, 2))


def calculate_lifecycle_emissions(
    fuel_type: str,
    fuel_tonnes: float,
    cargo_tonnes: float = 50000.0,
    distance_nm: float = 1000.0,
    bunker_price_usd: Optional[float] = None,
    include_carbon_tax: bool = False,
    carbon_tax_rate: float = 85.0,
    berth_hours: float = 0.0,
    auxiliary_power_kw: float = 1200.0,
    use_shore_power: bool = False,
    green_grid: bool = False
) -> Dict[str, Any]:
    """
    Computes rigorous Tank-to-Wake (TtW), Well-to-Tank (WtT), and Well-to-Wake (WtW)
    lifecycle GHG emissions along with operational costs and IMO CII rating.
    """
    spec = get_fuel_spec(fuel_type)
    fuel_mass_t = max(0.0, float(fuel_tonnes))
    unit_price = bunker_price_usd if (bunker_price_usd is not None and bunker_price_usd > 0) else spec["default_price_usd_tonne"]

    # 1. Tank-to-Wake Direct Combustion Emissions (tonnes)
    ttw_co2 = fuel_mass_t * spec["ttw_co2_factor"]
    ttw_ch4 = fuel_mass_t * spec["ttw_ch4_factor"]
    ttw_n2o = fuel_mass_t * spec["ttw_n2o_factor"]
    # Total TtW in CO2-equivalent:
    ttw_ghg_co2e = (ttw_co2 * GWP_CO2) + (ttw_ch4 * GWP_CH4) + (ttw_n2o * GWP_N2O)

    # 2. Well-to-Tank Upstream Lifecycle Emissions (tonnes CO2e)
    wtt_ghg_co2e = fuel_mass_t * spec["wtt_ghg_factor"]

    # 3. Well-to-Wake Total Lifecycle GHG (tonnes CO2e)
    voyage_wtw_co2e = ttw_ghg_co2e + wtt_ghg_co2e

    # 4. Port Berth / Shore Power Electrification (Cold Ironing) Accounting
    port_fuel_saved_t = 0.0
    port_co2e_saved_t = 0.0
    port_electricity_cost_usd = 0.0
    port_aux_diesel_cost_usd = 0.0
    port_net_cost_delta_usd = 0.0

    if berth_hours > 0 and auxiliary_power_kw > 0:
        total_port_energy_kwh = auxiliary_power_kw * berth_hours
        # Diesel auxiliary baseline:
        aux_diesel_tonnes = (total_port_energy_kwh * SHORE_POWER_DEFAULTS["marine_aux_diesel_sfoc_g_kwh"]) / 1_000_000.0
        aux_spec = get_fuel_spec("MGO")
        port_aux_diesel_co2e = aux_diesel_tonnes * (aux_spec["ttw_co2_factor"] + aux_spec["wtt_ghg_factor"])
        port_aux_diesel_cost_usd = aux_diesel_tonnes * aux_spec["default_price_usd_tonne"]

        if use_shore_power:
            grid_intensity = SHORE_POWER_DEFAULTS["green_grid_carbon_g_kwh"] if green_grid else SHORE_POWER_DEFAULTS["grid_carbon_intensity_g_kwh"]
            shore_co2e = (total_port_energy_kwh * grid_intensity) / 1_000_000.0
            port_electricity_cost_usd = total_port_energy_kwh * SHORE_POWER_DEFAULTS["electricity_tariff_usd_kwh"]
            
            port_fuel_saved_t = aux_diesel_tonnes
            port_co2e_saved_t = max(0.0, port_aux_diesel_co2e - shore_co2e)
            port_net_cost_delta_usd = port_electricity_cost_usd - port_aux_diesel_cost_usd
            # Adjust total lifecycle emissions
            voyage_wtw_co2e += shore_co2e
            ttw_ghg_co2e += 0.0  # Cold ironing produces zero local TtW emissions at berth
            wtt_ghg_co2e += shore_co2e
        else:
            voyage_wtw_co2e += port_aux_diesel_co2e
            ttw_ghg_co2e += (aux_diesel_tonnes * aux_spec["ttw_co2_factor"])
            wtt_ghg_co2e += (aux_diesel_tonnes * aux_spec["wtt_ghg_factor"])

    # 5. Financial Cost Breakdown
    direct_bunker_cost = fuel_mass_t * unit_price
    carbon_tax_cost = (voyage_wtw_co2e * carbon_tax_rate) if include_carbon_tax else 0.0
    total_fuel_cost_usd = direct_bunker_cost + carbon_tax_cost + (port_electricity_cost_usd if use_shore_power else port_aux_diesel_cost_usd)

    # 6. Carbon Intensity Indicator (CII) Proxy in g CO2e / (tonne * nautical mile)
    transport_work = max(1.0, cargo_tonnes * distance_nm)
    cii_value = (ttw_ghg_co2e * 1_000_000.0) / transport_work
    cii_rating, cii_color, cii_badge = get_cii_grade(cii_value)

    return {
        "fuel_type": fuel_type,
        "fuel_name": spec["name"],
        "category": spec["category"],
        "fuel_tonnes": round(fuel_mass_t, 2),
        "unit_price_usd_tonne": round(unit_price, 2),
        "lhv_mj_kg": spec["lhv_mj_kg"],
        
        # Emission breakdowns (tonnes)
        "ttw_co2_tonnes": round(ttw_co2, 2),
        "ttw_ch4_tonnes": round(ttw_ch4, 4),
        "ttw_n2o_tonnes": round(ttw_n2o, 5),
        "ttw_ghg_co2e_tonnes": round(ttw_ghg_co2e, 2),
        "wtt_ghg_co2e_tonnes": round(wtt_ghg_co2e, 2),
        "wtw_lifecycle_ghg_co2e_tonnes": round(voyage_wtw_co2e, 2),
        
        # Financial metrics (USD)
        "direct_bunker_cost_usd": round(direct_bunker_cost, 2),
        "carbon_tax_cost_usd": round(carbon_tax_cost, 2),
        "total_fuel_cost_usd": round(total_fuel_cost_usd, 2),
        
        # CII & Compliance
        "cii_value_g_co2e_t_nm": round(cii_value, 3),
        "cii_rating": cii_rating,
        "cii_color": cii_color,
        "cii_badge": cii_badge,
        
        # Cold ironing metrics
        "use_shore_power": use_shore_power,
        "port_fuel_saved_tonnes": round(port_fuel_saved_t, 2),
        "port_co2e_saved_tonnes": round(port_co2e_saved_t, 2),
        "port_electricity_cost_usd": round(port_electricity_cost_usd, 2),
        
        # Metadata
        "bunkering_availability": spec["bunkering_availability"],
        "regulatory_status": spec["regulatory_status"]
    }


def get_cii_grade(cii_value: float) -> tuple:
    """Returns grade letter, hex color, and descriptive badge for IMO Carbon Intensity Indicator."""
    if cii_value <= 5.0:
        return "A", "#10B981", "Rating A (Superior Green)"
    elif cii_value <= 7.5:
        return "B", "#34D399", "Rating B (Minor Superior)"
    elif cii_value <= 10.5:
        return "C", "#FBBF24", "Rating C (Compliant Baseline)"
    elif cii_value <= 14.0:
        return "D", "#F97316", "Rating D (Action Plan Required)"
    else:
        return "E", "#EF4444", "Rating E (Inferior / Non-Compliant)"


def compare_all_fuel_scenarios(
    baseline_fuel_tonnes: float,
    baseline_fuel_type: str = "VLSFO",
    cargo_tonnes: float = 50000.0,
    distance_nm: float = 1000.0,
    include_carbon_tax: bool = False,
    carbon_tax_rate: float = 85.0
) -> pd.DataFrame:
    """
    Evaluates all supported fuels on an iso-energetic basis (equal propulsive energy).
    Produces a comprehensive scenario comparison dataframe.
    """
    records = []
    for f_key in FUEL_DATABASE.keys():
        eq_tonnes = calculate_fuel_energy_mass(baseline_fuel_tonnes, baseline_fuel_type, f_key)
        res = calculate_lifecycle_emissions(
            fuel_type=f_key,
            fuel_tonnes=eq_tonnes,
            cargo_tonnes=cargo_tonnes,
            distance_nm=distance_nm,
            include_carbon_tax=include_carbon_tax,
            carbon_tax_rate=carbon_tax_rate
        )
        records.append({
            "Fuel Scenario": res["fuel_name"],
            "Category": res["category"],
            "Required Mass (t)": res["fuel_tonnes"],
            "LHV (MJ/kg)": res["lhv_mj_kg"],
            "Benchmark Price ($/t)": res["unit_price_usd_tonne"],
            "Bunker Cost ($)": res["direct_bunker_cost_usd"],
            "Tank-to-Wake (t CO2e)": res["ttw_ghg_co2e_tonnes"],
            "Well-to-Tank (t CO2e)": res["wtt_ghg_co2e_tonnes"],
            "Lifecycle GHG (t CO2e)": res["wtw_lifecycle_ghg_co2e_tonnes"],
            "CII Rating": res["cii_rating"],
            "Availability": res["bunkering_availability"]
        })
    return pd.DataFrame(records)
