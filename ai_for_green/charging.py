"""Continuous charging-power LP with independent post-solve auditing."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class Config:
    efficiency: float = .92
    slot_hours: float = 1.0
    bus_kw: float = 50.0
    depot_kw: float = 250.0
    battery_kwh: float = 240.0
    reserve_kwh: float = 20.0
    carbon_price: float = 2.0


def make_fleet(predictions, seed=42):
    last = predictions[predictions["date"] == predictions["date"].max()].copy()
    rng = np.random.default_rng(seed + 100)
    last["arrival_slot"] = rng.integers(0, 4, len(last))
    last["departure_slot"] = rng.integers(10, 13, len(last))
    last["arrival_energy_kwh"] = rng.uniform(45, 75, len(last))
    return last.reset_index(drop=True)


def inputs(fleet, grid, cfg):
    values = np.array([cfg.efficiency, cfg.slot_hours, cfg.bus_kw, cfg.depot_kw,
                       cfg.battery_kwh, cfg.reserve_kwh, cfg.carbon_price])
    if not np.isfinite(values).all() or not 0 < cfg.efficiency <= 1 or (values[1:5] <= 0).any() or (values[5:] < 0).any():
        raise ValueError("Invalid configuration")
    n, t = len(fleet), len(grid)
    if not n or not t:
        raise ValueError("Fleet and grid cannot be empty")
    arrival = fleet["arrival_slot"].to_numpy(float)
    departure = fleet["departure_slot"].to_numpy(float)
    energy = fleet["arrival_energy_kwh"].to_numpy(float)
    planning = fleet["planning_energy_kwh"].to_numpy(float)
    carbon = grid["carbon_kg_per_kwh"].to_numpy(float)
    price = grid["price_per_kwh"].to_numpy(float)
    if not all(np.isfinite(a).all() for a in [arrival, departure, energy, planning, carbon, price]):
        raise ValueError("All scheduling inputs must be finite")
    if ((arrival % 1 != 0) | (departure % 1 != 0) | (arrival < 0) | (departure > t) | (departure <= arrival)).any():
        raise ValueError("Invalid arrival/departure window")
    if ((energy < 0) | (energy > cfg.battery_kwh) | (planning < 0)).any() or (carbon < 0).any() or (price < 0).any():
        raise ValueError("Invalid energy, carbon intensity or price")
    target = np.maximum(energy, planning + cfg.reserve_kwh)
    if (target > cfg.battery_kwh + 1e-7).any():
        raise ValueError("Infeasible: planned departure energy exceeds battery capacity")
    demand = target - energy
    allowed = (np.arange(t)[None, :] >= arrival[:, None]) & (np.arange(t)[None, :] < departure[:, None])
    if (demand > allowed.sum(axis=1) * cfg.bus_kw * cfg.slot_hours * cfg.efficiency + 1e-7).any():
        raise ValueError("Infeasible: vehicle charging window too short")
    return demand, allowed, carbon, price


def solve_schedule(fleet, grid, strategy, cfg=Config()):
    demand, allowed, carbon, price = inputs(fleet, grid, cfg)
    n, t = allowed.shape
    objectives = {"immediate": np.arange(t, dtype=float), "cost": price,
                  "carbon": carbon, "balanced": price + cfg.carbon_price * carbon}
    if strategy not in objectives:
        raise ValueError(f"Unknown strategy: {strategy}")
    c = np.tile(objectives[strategy], n) * cfg.slot_hours
    energy_eq = np.zeros((n, n * t))
    depot_ub = np.zeros((t, n * t))
    for bus in range(n):
        energy_eq[bus, bus*t:(bus+1)*t] = cfg.efficiency * cfg.slot_hours
    for slot in range(t):
        depot_ub[slot, slot::t] = 1.0
    bounds = [(0, cfg.bus_kw if active else 0) for active in allowed.ravel()]
    result = linprog(c, A_eq=energy_eq, b_eq=demand, A_ub=depot_ub,
                     b_ub=np.full(t, cfg.depot_kw), bounds=bounds, method="highs")
    if not result.success:
        raise ValueError(f"Charging optimization failed: {result.message}")
    power = result.x.reshape(n, t)
    audit_schedule(fleet, grid, power, cfg)
    return power


def audit_schedule(fleet, grid, power, cfg=Config()):
    demand, allowed, carbon, price = inputs(fleet, grid, cfg)
    power = np.asarray(power, dtype=float)
    if power.shape != allowed.shape or not np.isfinite(power).all():
        raise ValueError("Invalid schedule shape or nonfinite power")
    tol = 1e-5
    charged = power.sum(axis=1) * cfg.slot_hours * cfg.efficiency
    residual = float(np.max(np.abs(charged - demand)))
    if power.min() < -tol or power.max() > cfg.bus_kw + tol or (power.sum(axis=0) > cfg.depot_kw + tol).any():
        raise ValueError("Schedule violates power limits")
    if (np.abs(power[~allowed]) > tol).any() or residual > tol:
        raise ValueError("Schedule violates availability or energy balance")
    departure_energy = fleet["arrival_energy_kwh"].to_numpy() + charged
    if (departure_energy > cfg.battery_kwh + tol).any():
        raise ValueError("Schedule violates battery capacity")
    grid_energy = power.sum(axis=0) * cfg.slot_hours
    actual = fleet["actual_energy_kwh"].to_numpy()
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("Invalid actual operational energy")
    reserve_gap = np.maximum(actual + cfg.reserve_kwh - departure_energy, 0)
    return {"grid_energy_kwh": float(grid_energy.sum()),
            "battery_energy_kwh": float(charged.sum()),
            "cost_units": float(grid_energy @ price),
            "carbon_kg": float(grid_energy @ carbon),
            "peak_kw": float(power.sum(axis=0).max()),
            "max_energy_error_kwh": residual,
            "buses_below_actual_reserve": int((reserve_gap > tol).sum()),
            "actual_reserve_gap_kwh": float(reserve_gap.sum())}


def schedule_frame(fleet, grid, plans):
    return pd.DataFrame([{"strategy": strategy, "bus_id": bus,
        "slot": slot, "time": grid.iloc[slot]["time"], "grid_kw": float(power[i, slot])}
        for strategy, power in plans.items() for i, bus in enumerate(fleet["bus_id"])
        for slot in range(len(grid))])
