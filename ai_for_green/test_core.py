import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
from charging import Config, solve_schedule, audit_schedule, make_fleet, inputs
from data import generate_history, validate_history
from forecast import date_splits, fit_forecast


class ChargingTests(unittest.TestCase):
    def setUp(self):
        self.fleet = pd.DataFrame({"arrival_slot": [0], "departure_slot": [2],
            "arrival_energy_kwh": [40.], "planning_energy_kwh": [50.], "actual_energy_kwh": [50.]})
        self.grid = pd.DataFrame({"carbon_kg_per_kwh": [1., .1], "price_per_kwh": [.1, 1.]})
        self.cfg = Config(efficiency=1., reserve_kwh=0., bus_kw=10., depot_kw=10.)

    def test_carbon_and_cost_have_known_opposite_optima(self):
        low_carbon = solve_schedule(self.fleet, self.grid, "carbon", self.cfg)
        low_cost = solve_schedule(self.fleet, self.grid, "cost", self.cfg)
        np.testing.assert_allclose(low_carbon, [[0, 10]], atol=1e-7)
        np.testing.assert_allclose(low_cost, [[10, 0]], atol=1e-7)
        self.assertAlmostEqual(audit_schedule(self.fleet, self.grid, low_carbon, self.cfg)["carbon_kg"], 1.)

    def test_no_charging_before_arrival_and_efficiency(self):
        self.fleet["arrival_slot"] = 1
        self.fleet["planning_energy_kwh"] = 45
        cfg = Config(efficiency=.5, reserve_kwh=0., bus_kw=10., depot_kw=10.)
        power = solve_schedule(self.fleet, self.grid, "immediate", cfg)
        np.testing.assert_allclose(power, [[0, 10]], atol=1e-7)
        self.assertAlmostEqual(audit_schedule(self.fleet, self.grid, power, cfg)["battery_energy_kwh"], 5.)

    def test_window_infeasible(self):
        self.fleet["planning_energy_kwh"] = 65
        with self.assertRaisesRegex(ValueError, "window"):
            solve_schedule(self.fleet, self.grid, "carbon", self.cfg)

    def test_depot_infeasible(self):
        fleet = pd.concat([self.fleet] * 3, ignore_index=True)
        with self.assertRaisesRegex(ValueError, "failed"):
            solve_schedule(fleet, self.grid, "carbon", self.cfg)

    def test_capacity_infeasible_not_silently_clipped(self):
        self.fleet["planning_energy_kwh"] = 250
        with self.assertRaisesRegex(ValueError, "capacity"):
            solve_schedule(self.fleet, self.grid, "carbon", self.cfg)

    def test_nonfinite_input(self):
        self.grid.loc[0, "carbon_kg_per_kwh"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            solve_schedule(self.fleet, self.grid, "carbon", self.cfg)

    def test_audit_rejects_energy_shortfall(self):
        with self.assertRaisesRegex(ValueError, "energy balance"):
            audit_schedule(self.fleet, self.grid, np.zeros((1, 2)), self.cfg)

    def test_actual_reserve_gap_reported(self):
        self.fleet["actual_energy_kwh"] = 55
        power = solve_schedule(self.fleet, self.grid, "carbon", self.cfg)
        metric = audit_schedule(self.fleet, self.grid, power, self.cfg)
        self.assertEqual(metric["buses_below_actual_reserve"], 1)
        self.assertAlmostEqual(metric["actual_reserve_gap_kwh"], 5.)


class DataTests(unittest.TestCase):
    def test_default_demo_fits_physical_scenario(self):
        result = fit_forecast(generate_history())
        fleet = make_fleet(result['predictions'])
        from data import generate_grid
        inputs(fleet, generate_grid(), Config())

    def test_reproducible_and_date_disjoint(self):
        history = generate_history(30, 3, 42)
        pd.testing.assert_frame_equal(history, generate_history(30, 3, 42))
        train, calibration, test = date_splits(history)
        self.assertLess(train.date.max(), calibration.date.min())
        self.assertLess(calibration.date.max(), test.date.min())
        self.assertEqual(len(train) + len(calibration) + len(test), len(history))

    def test_incomplete_fleet_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete fleet"):
            validate_history(generate_history(30, 3).iloc[1:])


if __name__ == "__main__":
    unittest.main()
