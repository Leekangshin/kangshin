"""Save inspectable tables, metrics, and a headless four-panel chart."""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"immediate": "#738194", "cost": "#3478bd", "carbon": "#159d87", "balanced": "#eea13b"}


def write_report(output, forecast, fleet, grid, plans, metrics, config, source, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pred = forecast["predictions"]
    pred.to_csv(output / "test_predictions.csv", index=False)
    fleet.to_csv(output / "departure_requirements.csv", index=False)
    grid.to_csv(output / "grid_profile.csv", index=False)
    from charging import schedule_frame
    schedule_frame(fleet, grid, plans).to_csv(output / "charging_schedule.csv", index=False)
    pd.DataFrame(metrics).T.rename_axis("strategy").to_csv(output / "strategy_metrics.csv")
    package = {"history_source": source, "grid_source": "synthetic",
        "seed": seed, "config": config.__dict__, "forecast": forecast["metrics"],
        "feature_importance": forecast["feature_importance"], "strategies": metrics,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}}
    (output / "metrics.json").write_text(json.dumps(package, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#f4f7fa"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("AI FOR GREEN  |  ELECTRIC BUS CHARGING\nSynthetic demonstration · same energy demand in every strategy", fontsize=16, fontweight="bold")
    ax = axes[0, 0]
    ax.scatter(pred["actual_energy_kwh"], pred["predicted_energy_kwh"], s=10, alpha=.45, color=COLORS["carbon"])
    low, high = pred["actual_energy_kwh"].min(), pred["actual_energy_kwh"].max()
    ax.plot([low, high], [low, high], "--", color="#8693a0", linewidth=1)
    ax.set(xlabel="Observed battery energy (kWh)", ylabel="AI prediction (kWh)",
           title=f"A  Chronological test · MAE {forecast['metrics']['random_forest']['mae_kwh']:.2f} kWh")
    ax = axes[0, 1]
    ax.plot(grid["slot"], grid["carbon_kg_per_kwh"], marker="o", color=COLORS["carbon"], label="Carbon intensity")
    ax.set(ylabel="Average carbon intensity (kgCO2/kWh)", title="B  Synthetic overnight electricity profile")
    twin = ax.twinx()
    twin.plot(grid["slot"], grid["price_per_kwh"], marker="s", linestyle="--", color=COLORS["cost"], label="Price")
    twin.set_ylabel("Price (currency units/kWh)")
    ax.set_xticks(grid["slot"][::2], grid["time"][::2])
    ax.legend(loc="upper right", fontsize=8)
    twin.legend(loc="lower left", fontsize=8)
    ax = axes[1, 0]
    for strategy, power in plans.items():
        ax.step(grid["slot"], power.sum(axis=0), where="mid", color=COLORS[strategy], label=strategy, linewidth=2)
    ax.axhline(config.depot_kw, color="#bb3b3b", linestyle=":", label="Depot limit")
    ax.set_xticks(grid["slot"][::2], grid["time"][::2])
    ax.set(ylabel="Depot grid power (kW)", title="C  Four feasible charging schedules")
    ax.legend(ncol=2, fontsize=8)
    ax = axes[1, 1]
    for strategy, metric in metrics.items():
        ax.scatter(metric["cost_units"], metric["carbon_kg"], s=100,
                   facecolors="none" if strategy == "balanced" else COLORS[strategy],
                   edgecolors=COLORS[strategy], linewidths=2, zorder=4)
        offset = {"immediate": (6, 8), "cost": (-30, 12),
                  "carbon": (10, -18), "balanced": (10, 12)}[strategy]
        ax.annotate(strategy, (metric["cost_units"], metric["carbon_kg"]), xytext=offset,
                    textcoords="offset points", fontsize=9)
    ax.margins(.25)
    ax.set(xlabel="Electricity cost (currency units)", ylabel="Charging emissions (kgCO2)",
           title="D  Cost and charging-carbon trade-off")
    for ax in axes.ravel():
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    rows = ["| 策略 | 电网输入 kWh | 电费（货币单位） | 充电 kgCO2 | 相对尽早充电减排 | 储备不足车辆 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in metrics.items():
        reduction = 100 * (1 - m["carbon_kg"] / metrics["immediate"]["carbon_kg"])
        rows.append(f"| {name} | {m['grid_energy_kwh']:.2f} | {m['cost_units']:.2f} | {m['carbon_kg']:.2f} | {reduction:.2f}% | {m['buses_below_actual_reserve']} |")
    f = forecast["metrics"]
    report = f'''# 电动公交低碳充电：实际运行报告

本报告为模拟演示。历史数据来源：`{source}`；电价、碳强度、到场时间与电量均为模拟值。数值不是现实公交场站减排承诺。

## 预测评估

- 日期按60%/20%/20%划分为训练、校准、测试；同一日期的所有车辆位于同一子集。
- 随机森林测试 MAE：{f['random_forest']['mae_kwh']:.3f} kWh；RMSE：{f['random_forest']['rmse_kwh']:.3f} kWh。
- 里程基线测试 MAE：{f['distance_baseline']['mae_kwh']:.3f} kWh；RMSE：{f['distance_baseline']['rmse_kwh']:.3f} kWh。
- 校准残差90%分位余量：{f['calibration_margin_kwh']:.3f} kWh；测试集经验覆盖率：{100*f['test_margin_coverage']:.2f}%。此值不是保证。
- 计划基于最后一个测试日的{len(fleet)}辆公交。上一日真实用电是逐日预测时已观测的特征，不支持一次性预测所有未来日期。

## 充电比较

''' + "\n".join(rows) + f'''

所有策略有相同的车辆电量目标、时间窗、充电效率和功率限制。单车上限{config.bus_kw:g} kW，场站上限{config.depot_kw:g} kW。
`immediate`是最小化充电时刻加权能量的受约束尽早充电基准；`cost`只优化电费；`carbon`只优化碳；`balanced`优化电费+{config.carbon_price:g}×kgCO2。
折中目标不保证产生不同的方案；本默认场景的折中方案与最低碳方案在总指标上相同。
每辆车有独立接入接口，功率可以连续变化；不模拟有限充电枪、离散开关、车辆调位、电池老化和充电曲线。

碳排放边界为电网平均碳强度对应的充电用电，不包含电池/车辆生产，不等于边际排放减少量。电费仅含kWh电量费，不含需量费。
车辆需求超过容量、充电窗口或场站可行性时会明确报错；不会少充电来制造收益。

## 核验与文件

优化完成后独立检查每车时间窗、单车及场站功率、充入能量和电池容量。真实运行储备缺口另外计算，见表中车辆数量及metrics.json。

![对比图](comparison.png)

- `test_predictions.csv`：时间测试集逐车预测与真实用电。
- `departure_requirements.csv`：最后一天的出发要求与到场电量。
- `charging_schedule.csv`：逐策略、车辆、小时的电网kW。
- `strategy_metrics.csv` / `metrics.json`：评估结果、参数、种子和实际库版本。

基础参考：[NREL受控充电综述](https://www.nrel.gov/news/program/2022/aligning-utilities-electric-vehicles-for-greater-grid.html)、
[时间序列评估](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)、
[SciPy HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ipm.html)。这些文献提供背景和工具依据，不验证本演示的结果百分比。
'''
    (output / "REPORT.md").write_text(report, encoding="utf-8")
