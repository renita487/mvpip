"""
========================================================================================
EGREEN QUANTA — UTILITIES & SCIENTIFIC ENGINE MODULES
SIH26138 — Quantum-Inspired Green Fleet Management
========================================================================================
"""

from utils.fuel_scenarios import (
    FUEL_DATABASE,
    get_fuel_spec,
    calculate_fuel_energy_mass,
    calculate_lifecycle_emissions,
    compare_all_fuel_scenarios,
    get_cii_grade
)

from utils.physics_engine import (
    VESSEL_SPECS,
    WEATHER_FACTORS,
    calculate_hydrodynamic_fuel,
    compute_slow_steaming_comparison,
    generate_speed_curve_data
)

from utils.constraint_engine import (
    FleetConstraintEngine,
    default_constraint_engine
)

from utils.ml_surrogate import (
    MarineSurrogatePipeline,
    get_ml_surrogate_engine,
    load_maritime_dataset
)

from utils.qpso_optimizer import (
    DEFAULT_FLEET,
    run_qpso_optimization,
    run_classical_pso_optimization,
    run_full_benchmark,
    evaluate_fleet_plan,
    decode_particle_to_fleet_plan,
    extract_pareto_tradeoffs
)
