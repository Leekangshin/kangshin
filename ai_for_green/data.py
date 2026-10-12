"""Reproducible synthetic bus operations. No real operator data is bundled."""
import numpy as np
import pandas as pd

FEATURES = ["planned_km", "temperature_forecast_c", "congestion_forecast",
            "passenger_forecast", "hill_index", "weekday", "lag_energy_kwh"]
REQUIRED = ["date", "bus_id", *FEATURES, "actual_energy_kwh"]


def generate_history(days=120, buses=16, seed=42):
    if days < 30 or buses < 1:
        raise ValueError("Need at least 30 days and one bus")
    rng = np.random.default_rng(seed)
    hills = rng.uniform(0, 1, buses)
    previous = np.full(buses, 135.0)
    rows = []
    for day, date in enumerate(pd.date_range("2026-01-01", periods=days)):
        # Constructed weather cycle, covered by the training interval.
        # This demo is not a seasonal distribution-shift benchmark.
        temperature = 15 + 12 * np.sin(day / 9) + rng.normal(0, 2)
        for bus in range(buses):
            km = rng.uniform(85, 125) * (0.9 if date.weekday() >= 5 else 1)
            congestion = rng.uniform(0.1, 0.9)
            passengers = rng.uniform(12, 55)
            # Actual outcomes are not supplied as prediction features.
            real_temp = temperature + rng.normal(0, 1.4)
            rate = (1.05 + 0.24 * hills[bus] + 0.20 * congestion
                    + 0.003 * passengers + 0.003 * (real_temp - 20) ** 2)
            energy = max(5, km * rate + rng.normal(0, 3.5))
            rows.append(dict(date=date.strftime("%Y-%m-%d"), bus_id=f"BUS-{bus+1:02d}",
                planned_km=km, temperature_forecast_c=temperature,
                congestion_forecast=congestion, passenger_forecast=passengers,
                hill_index=hills[bus], weekday=date.weekday(),
                lag_energy_kwh=previous[bus], actual_energy_kwh=energy,
                data_source="synthetic"))
            previous[bus] = energy
    return validate_history(pd.DataFrame(rows))


def validate_history(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any() or frame["bus_id"].isna().any():
        raise ValueError("Date and bus_id must be present")
    if frame.duplicated(["date", "bus_id"]).any():
        raise ValueError("Duplicate date/bus_id rows")
    if frame["date"].nunique() < 30:
        raise ValueError("Need at least 30 complete dates for chronological evaluation")
    numeric = FEATURES + ["actual_energy_kwh"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Numeric inputs must be finite")
    if (frame[["planned_km", "actual_energy_kwh"]] <= 0).any().any():
        raise ValueError("Distance and energy must be positive")
    if (frame[["passenger_forecast", "lag_energy_kwh"]] < 0).any().any():
        raise ValueError("Passengers and lag energy cannot be negative")
    if not frame["congestion_forecast"].between(0, 1).all() or not frame["hill_index"].between(0, 1).all():
        raise ValueError("Congestion and hill_index must be in [0, 1]")
    if not np.array_equal(frame["weekday"].to_numpy(), frame["date"].dt.weekday.to_numpy()):
        raise ValueError("weekday must match date (Monday=0)")
    all_buses = set(frame["bus_id"])
    if any(set(group["bus_id"]) != all_buses for _, group in frame.groupby("date")):
        raise ValueError("Each date must contain the same complete fleet")
    return frame.sort_values(["date", "bus_id"]).reset_index(drop=True)


def generate_grid():
    return pd.DataFrame({"slot": np.arange(12),
        "time": [f"{hour % 24:02d}:00" for hour in range(18, 30)],
        "carbon_kg_per_kwh": [.64, .70, .67, .61, .50, .46, .38, .29, .25, .30, .42, .55],
        "price_per_kwh": [.90, .92, .84, .70, .50, .38, .32, .35, .42, .50, .78, .88],
        "data_source": "synthetic"})
