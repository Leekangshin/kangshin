# 电动公交低碳充电：实际运行报告

本报告为模拟演示。历史数据来源：`synthetic`；电价、碳强度、到场时间与电量均为模拟值。数值不是现实公交场站减排承诺。

## 预测评估

- 日期按60%/20%/20%划分为训练、校准、测试；同一日期的所有车辆位于同一子集。
- 随机森林测试 MAE：10.699 kWh；RMSE：13.796 kWh。
- 里程基线测试 MAE：30.398 kWh；RMSE：38.089 kWh。
- 校准残差90%分位余量：13.016 kWh；测试集经验覆盖率：85.94%。此值不是保证。
- 计划基于最后一个测试日的16辆公交。上一日真实用电是逐日预测时已观测的特征，不支持一次性预测所有未来日期。

## 充电比较

| 策略 | 电网输入 kWh | 电费（货币单位） | 充电 kgCO2 | 相对尽早充电减排 | 储备不足车辆 |
| --- | ---: | ---: | ---: | ---: | ---: |
| immediate | 2109.67 | 1273.56 | 1089.92 | 0.00% | 0 |
| cost | 2109.67 | 1079.62 | 875.98 | 19.63% | 0 |
| carbon | 2109.67 | 1084.20 | 863.69 | 20.76% | 0 |
| balanced | 2109.67 | 1084.20 | 863.69 | 20.76% | 0 |

所有策略有相同的车辆电量目标、时间窗、充电效率和功率限制。单车上限50 kW，场站上限250 kW。
`immediate`是最小化充电时刻加权能量的受约束尽早充电基准；`cost`只优化电费；`carbon`只优化碳；`balanced`优化电费+2×kgCO2。
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
