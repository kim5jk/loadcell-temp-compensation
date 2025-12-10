# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Load Cell Temperature Compensation - Basic Usage
# 
# This notebook demonstrates how to use the `loadcell-temp-compensation` library
# to improve measurement stability from consumer-grade load cells.

# %% 
# Import libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the compensator
from loadcell_temp_comp import TemperatureCompensator, compensate_temperature

# %% [markdown]
# ## Load Sample Data
# 
# Your data should have at minimum:
# - Timestamps (datetime)
# - Weight measurements (grams)
# - Temperature measurements (°C)

# %%
# Example: Load your data
# df = pd.read_csv('your_data.csv')
# df['datetime'] = pd.to_datetime(df['datetime'])

# For this example, let's create synthetic data
np.random.seed(42)
n_points = 500
hours = np.linspace(0, 48, n_points)  # 48 hours of data

# Simulate temperature variation (daily cycle)
temperature = 18 + 4 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.3, n_points)

# Simulate weight with temperature drift (~2 g/°C sensitivity)
true_weight = 350  # grams
temp_drift = 2.0 * (temperature - 18)  # 2 g/°C
noise = np.random.normal(0, 0.2, n_points)
weight = true_weight + temp_drift + noise

# Add a consumption event at hour 24
consumption_idx = n_points // 2
weight[consumption_idx:] -= 15  # 15g removed

df = pd.DataFrame({
    'datetime': pd.date_range('2024-01-01', periods=n_points, freq='6min'),
    'weight_g': weight,
    'temperature_c': temperature
})

print(f"Data shape: {df.shape}")
print(f"Temperature range: {df['temperature_c'].min():.1f}°C to {df['temperature_c'].max():.1f}°C")
print(f"Weight range: {df['weight_g'].min():.1f}g to {df['weight_g'].max():.1f}g")

# %% [markdown]
# ## Method 1: Quick Compensation (One-liner)

# %%
# Simple one-liner for quick results
result = compensate_temperature(
    timestamps=df['datetime'],
    weights=df['weight_g'],
    temperatures=df['temperature_c']
)

print(f"Raw weight std: {df['weight_g'].std():.3f}g")
print(f"Compensated weight std: {result['adjusted_weight'].std():.3f}g")

# %% [markdown]
# ## Method 2: Full Control with TemperatureCompensator Class

# %%
# Initialize with custom parameters
compensator = TemperatureCompensator(
    sigma_temp=30,           # Smoothing for temperature
    sigma_weight=30,         # Smoothing for weight
    drift_threshold=0.4,     # Threshold for segment detection (g)
    reference_temperature=20.0,  # Normalize to this temp
    detrend_threshold=0.5    # Long-term drift threshold (g/day)
)

print(compensator)

# %%
# Fit and transform
result = compensator.fit_transform(
    timestamps=df['datetime'],
    weights=df['weight_g'],
    temperatures=df['temperature_c']
)

# View results
print("\nResult columns:", result.columns.tolist())
result.head()

# %%
# Get per-segment statistics
segment_stats = compensator.get_segment_stats()
print(f"\nAnalyzed {len(segment_stats)} segments")
segment_stats

# %%
# Get overall summary
summary = compensator.get_summary()
print("\n=== Compensation Summary ===")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")

# %% [markdown]
# ## Visualization

# %%
# Create comparison plot
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Raw weight
fig.add_trace(
    go.Scatter(x=result['timestamp'], y=result['weight'], 
               mode='markers', name='Raw Weight',
               marker=dict(color='blue', size=4, opacity=0.5)),
    secondary_y=False,
)

# Compensated weight
fig.add_trace(
    go.Scatter(x=result['timestamp'], y=result['adjusted_weight'], 
               mode='lines', name='Compensated Weight',
               line=dict(color='green', width=2)),
    secondary_y=False,
)

# Temperature
fig.add_trace(
    go.Scatter(x=result['timestamp'], y=result['smoothed_temp'], 
               mode='lines', name='Temperature',
               line=dict(color='lightblue', width=1.5)),
    secondary_y=True,
)

fig.update_layout(
    title='Temperature Compensation Results',
    xaxis_title='Time',
    template='plotly_white',
    height=500,
)
fig.update_yaxes(title_text="Weight (g)", secondary_y=False)
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

fig.show()

# %% [markdown]
# ## Calculate Improvement Metrics

# %%
# Per-segment analysis
print("=== Per-Segment Improvement ===\n")
for _, row in segment_stats.iterrows():
    print(f"Segment {row['segment_id']}: {row['improvement_pct']:.1f}% improvement "
          f"(raw: {row['raw_std']:.3f}g → adj: {row['adjusted_std']:.3f}g)")

# Overall
print(f"\n=== Overall Results ===")
print(f"Mean raw std: {segment_stats['raw_std'].mean():.3f}g")
print(f"Mean compensated std: {segment_stats['adjusted_std'].mean():.3f}g")
print(f"Improvement factor: {segment_stats['raw_std'].mean() / segment_stats['adjusted_std'].mean():.1f}x")

# %% [markdown]
# ## Tips for Best Results
# 
# 1. **Adjust `sigma_temp` and `sigma_weight`**: Higher values = more smoothing. 
#    Start with 30, increase if data is noisy.
# 
# 2. **Tune `drift_threshold`**: Set based on your expected consumption events.
#    Too low = many small segments. Too high = missed events.
# 
# 3. **Reference temperature**: Set to your typical operating temperature.
# 
# 4. **Data quality**: Ensure timestamps are monotonic and no missing values.
