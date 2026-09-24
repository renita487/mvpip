import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

np.random.seed(42)
n_samples = 2000

vessel_types = ['Container', 'Bulk Carrier', 'Tanker', 'General Cargo']
weather_conditions = ['Calm', 'Moderate', 'Rough']
fuel_types = ['HFO', 'LNG', 'Methanol', 'MGO']

data = {
    'vessel_id': [f'V{str(i).zfill(4)}' for i in range(1, n_samples+1)],
    'vessel_type': np.random.choice(vessel_types, n_samples),
    'capacity_dwt': np.random.randint(5000, 200000, n_samples),
    'speed_knots': np.round(np.random.uniform(10, 22, n_samples), 1),
    'cargo_load_tonnes': np.random.randint(3000, 180000, n_samples),
    'distance_nm': np.random.randint(200, 8000, n_samples),
    'weather': np.random.choice(weather_conditions, n_samples, p=[0.5, 0.35, 0.15]),
    'wind_speed_knots': np.round(np.random.uniform(0, 35, n_samples), 1),
    'wave_height_m': np.round(np.random.uniform(0.5, 6.0, n_samples), 1),
    'fuel_type': np.random.choice(fuel_types, n_samples, p=[0.55, 0.25, 0.1, 0.1]),
    'engine_power_kw': np.random.randint(3000, 60000, n_samples),
}

df = pd.DataFrame(data)

# Realistic fuel consumption formula
base_consumption = (
    0.0008 * df['speed_knots']**3 * (df['cargo_load_tonnes']/10000) * (df['distance_nm']/100) +
    0.15 * df['engine_power_kw']/1000
)

weather_factor = df['weather'].map({'Calm': 1.0, 'Moderate': 1.15, 'Rough': 1.35})
wind_factor = 1 + (df['wind_speed_knots'] / 100)
wave_factor = 1 + (df['wave_height_m'] / 20)
fuel_factor = df['fuel_type'].map({'HFO': 1.0, 'LNG': 0.92, 'Methanol': 0.95, 'MGO': 0.97})

df['fuel_consumption_tonnes'] = np.round(
    base_consumption * weather_factor * wind_factor * wave_factor * fuel_factor * np.random.uniform(0.9, 1.1, n_samples), 2
)

df['co2_emissions_tonnes'] = np.round(df['fuel_consumption_tonnes'] * 3.17, 2)

# Reorder columns
df = df[['vessel_id', 'vessel_type', 'capacity_dwt', 'speed_knots', 'cargo_load_tonnes', 
         'distance_nm', 'weather', 'wind_speed_knots', 'wave_height_m', 'fuel_type', 
         'engine_power_kw', 'fuel_consumption_tonnes', 'co2_emissions_tonnes']]

# Save the dataset
df.to_csv('ship_fuel_dataset.csv', index=False)
print("Dataset created successfully! File name: ship_fuel_dataset.csv")
print(df.head())