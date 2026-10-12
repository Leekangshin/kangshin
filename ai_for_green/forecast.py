"""Daily ahead forecast; all buses on one date stay in the same split."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data import FEATURES, validate_history


def date_splits(history):
    dates = np.sort(history["date"].unique())
    n_train, n_cal = int(len(dates) * .6), int(len(dates) * .2)
    return [history[history["date"].isin(part)].copy() for part in
            (dates[:n_train], dates[n_train:n_train+n_cal], dates[n_train+n_cal:])]


def fit_forecast(history, seed=42):
    history = validate_history(history)
    train, calibration, test = date_splits(history)
    model = RandomForestRegressor(n_estimators=180, min_samples_leaf=3,
                                  random_state=seed, n_jobs=1)
    model.fit(train[FEATURES], train["actual_energy_kwh"])
    prediction = model.predict(test[FEATURES])
    baseline_rate = train["actual_energy_kwh"].sum() / train["planned_km"].sum()
    baseline = test["planned_km"].to_numpy() * baseline_rate
    residual = calibration["actual_energy_kwh"].to_numpy() - model.predict(calibration[FEATURES])
    margin = max(0, float(np.quantile(residual, .9)))
    test["predicted_energy_kwh"] = prediction
    test["baseline_energy_kwh"] = baseline
    test["planning_energy_kwh"] = prediction + margin
    metrics = {}
    for name, pred in [("random_forest", prediction), ("distance_baseline", baseline)]:
        metrics[name] = {"mae_kwh": float(mean_absolute_error(test["actual_energy_kwh"], pred)),
            "rmse_kwh": float(np.sqrt(mean_squared_error(test["actual_energy_kwh"], pred)))}
    metrics["test_margin_coverage"] = float(np.mean(test["actual_energy_kwh"] <= prediction + margin))
    metrics["calibration_margin_kwh"] = margin
    metrics["splits"] = {name: {"start": str(part["date"].min().date()),
        "end": str(part["date"].max().date()), "rows": len(part)}
        for name, part in zip(["train", "calibration", "test"], [train, calibration, test])}
    return {"predictions": test, "metrics": metrics,
            "feature_importance": dict(zip(FEATURES, model.feature_importances_.tolist()))}
