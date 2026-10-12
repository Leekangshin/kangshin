"""python run_demo.py --output results"""
import argparse
from pathlib import Path
import pandas as pd
from data import generate_history, generate_grid, validate_history
from forecast import fit_forecast
from charging import Config, make_fleet, solve_schedule, audit_schedule
from reporting import write_report


def main():
    parser = argparse.ArgumentParser(description="AI for Green: synthetic electric bus charging")
    parser.add_argument("--data", type=Path, help="Optional history CSV with documented columns")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--buses", type=int, default=16)
    parser.add_argument("--carbon-price", type=float, default=2.0, help="Currency units/kgCO2 for balanced objective")
    args = parser.parse_args()
    history = validate_history(pd.read_csv(args.data, float_precision="round_trip")) if args.data else generate_history(args.days, args.buses, args.seed)
    source = "external_csv_unverified" if args.data else "synthetic"
    forecast = fit_forecast(history, args.seed)
    fleet, grid = make_fleet(forecast["predictions"], args.seed), generate_grid()
    config = Config(carbon_price=args.carbon_price)
    plans = {name: solve_schedule(fleet, grid, name, config) for name in ["immediate", "cost", "carbon", "balanced"]}
    metrics = {name: audit_schedule(fleet, grid, plan, config) for name, plan in plans.items()}
    write_report(args.output, forecast, fleet, grid, plans, metrics, config, source, args.seed)
    example_dir = args.output / "input_data"
    example_dir.mkdir(exist_ok=True)
    history.to_csv(example_dir / "fleet_history.csv", index=False)
    print(f"History: {source}; grid profile: synthetic; seed: {args.seed}")
    print(f"Forecast MAE: {forecast['metrics']['random_forest']['mae_kwh']:.3f} kWh")
    for name, m in metrics.items():
        print(f"{name:10} grid={m['grid_energy_kwh']:.2f} kWh; cost={m['cost_units']:.2f}; carbon={m['carbon_kg']:.2f} kgCO2; reserve-shortfall buses={m['buses_below_actual_reserve']}")
    print(f"Report: {(args.output / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
