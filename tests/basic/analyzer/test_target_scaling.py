import pytest
import pandas as pd
import numpy as np
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm
from tablemage._src.data.preprocessing import (
    CombinedSingleVarScaler,
    LogTransformSingleVar,
    Log1PTransformSingleVar,
    MinMaxSingleVar,
    StandardizeSingleVar,
)


@pytest.fixture
def setup_data():
    y = np.arange(20)
    y[0] = 1
    y = y * 100

    df_simple = pd.DataFrame({"y": y, "x1": np.arange(20), "x2": np.arange(20)[::-1]})

    return {"df_simple": df_simple}


def test_ols_scaling_simple(setup_data):
    """Tests minmax and standard scaling on a simple dataset."""

    df_simple = setup_data["df_simple"]

    analyzer = tm.Analyzer(df_simple, test_size=0.2, verbose=False)

    # quick minmax test
    rmse_unscaled = (
        analyzer.ols(target="y").metrics("test").loc["rmse", "OLS Linear Model"]
    )

    analyzer.scale(strategy="minmax")

    report = analyzer.ols(target="y")
    rmse_scaled = report.metrics("test").loc["rmse", "OLS Linear Model"]

    scaler = report.model()._dataemitter.y_scaler()
    assert len(scaler) == 1
    assert isinstance(scaler, CombinedSingleVarScaler)
    assert isinstance(scaler.scalers[0], MinMaxSingleVar)

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    # once we reset the data, we should remove the y scaler
    analyzer.load_data_checkpoint()

    report = analyzer.ols(target="y")
    rmse_unscaled_2 = report.metrics("test").loc["rmse", "OLS Linear Model"]

    assert report.model()._dataemitter.y_scaler() is None

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_unscaled_2)

    # quick standardize test
    analyzer.scale(strategy="standardize")
    report = analyzer.ols(target="y")
    rmse_scaled = report.metrics("test").loc["rmse", "OLS Linear Model"]
    scaler = report.model()._dataemitter.y_scaler()
    assert len(scaler) == 1
    assert isinstance(scaler, CombinedSingleVarScaler)
    assert isinstance(scaler.scalers[0], StandardizeSingleVar)
    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    # quick dual scale test
    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="standardize")
    analyzer.scale(include_vars=["x1", "y"], strategy="minmax")
    yscaler = analyzer.datahandler().scaler("y")
    assert isinstance(yscaler, CombinedSingleVarScaler)
    assert isinstance(yscaler.scalers[0], StandardizeSingleVar)
    assert isinstance(yscaler.scalers[1], MinMaxSingleVar)
    report = analyzer.ols(target="y")
    rmse_scaled = report.metrics("test").loc["rmse", "OLS Linear Model"]
    yscaler = report.model()._dataemitter.y_scaler()
    assert isinstance(yscaler, CombinedSingleVarScaler)
    assert isinstance(yscaler.scalers[0], StandardizeSingleVar)
    assert isinstance(yscaler.scalers[1], MinMaxSingleVar)
    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)


def test_ols_scaling_log(setup_data):
    """Tests log scaling on a simple dataset."""

    df_simple = setup_data["df_simple"]

    analyzer = tm.Analyzer(df_simple, test_size=0.2, verbose=False)

    report = analyzer.ols(target="y")
    assert report.model()._dataemitter.y_scaler() is None

    analyzer.scale(include_vars=["y"], strategy="log")

    report = analyzer.ols(target="y")
    yscaler = report.model()._dataemitter.y_scaler()
    assert isinstance(yscaler, CombinedSingleVarScaler)
    assert isinstance(yscaler.scalers[0], LogTransformSingleVar)

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="log1p")
    report = analyzer.ols(target="y")
    yscaler = report.model()._dataemitter.y_scaler()
    assert isinstance(yscaler, CombinedSingleVarScaler)
    assert isinstance(yscaler.scalers[0], Log1PTransformSingleVar)


def test_ml_scaling_simple(setup_data):
    """Tests minmax and standard scaling on a simple dataset."""

    df_simple = setup_data["df_simple"]

    analyzer = tm.Analyzer(df_simple, test_size=0.2, verbose=False)

    # quick minmax test
    report = analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
    rmse_unscaled = report.metrics("test").loc["rmse", "ols"]
    assert report.model("ols")._dataemitter.y_scaler() is None

    analyzer.scale(strategy="minmax")
    report = analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
    yscaler = report.model("ols")._dataemitter.y_scaler()
    assert isinstance(yscaler, CombinedSingleVarScaler)
    assert isinstance(yscaler.scalers[0], MinMaxSingleVar)
    rmse_scaled = report.metrics("test").loc["rmse", "ols"]

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    # once we reset the data, we should remove the y scaler
    analyzer.load_data_checkpoint()
    rmse_unscaled_2 = (
        analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
        .metrics("test")
        .loc["rmse", "ols"]
    )

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_unscaled_2)

    # quick standardize test
    analyzer.scale(strategy="standardize")
    rmse_scaled = (
        analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
        .metrics("test")
        .loc["rmse", "ols"]
    )
    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="log")
    report = analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
    yscaler = report.model("ols")._dataemitter.y_scaler()
    assert isinstance(yscaler, CombinedSingleVarScaler)
    assert isinstance(yscaler.scalers[0], LogTransformSingleVar)

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="log1p")
    report = analyzer.regress(
        models=[tm.ml.LinearR(name="ols"), tm.ml.TreesR(name="tree", n_trials=3)],
        target="y",
    )
    assert isinstance(
        report.model("ols")._dataemitter.y_scaler(), CombinedSingleVarScaler
    )
    assert isinstance(
        report.model("tree")._dataemitter.y_scaler(), CombinedSingleVarScaler
    )
