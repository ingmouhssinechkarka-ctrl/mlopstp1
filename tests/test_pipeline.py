import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import train
from train import FEATURES, TARGET, load_data, train as run_training

DATA = train.ROOT / "data" / "iris.csv"


# ---------- Données ----------

def test_dataset_schema():
    df = pd.read_csv(DATA)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 150
    assert df.isna().sum().sum() == 0


def test_load_data_classes_balanced():
    X, y = load_data(DATA)
    assert X.shape == (150, 4)
    assert sorted(y.unique()) == [0, 1, 2]
    assert y.value_counts().min() == 50


def test_load_data_rejects_bad_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_data(bad)


# ---------- Modèle ----------

@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    import mlflow
    tmp = tmp_path_factory.mktemp("run")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")
    out = tmp / "model"
    run_training(DATA, out, experiment="test")
    return out


def test_training_exports_model(model_dir):
    assert (model_dir / "model.joblib").exists()
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["features"] == FEATURES
    assert meta["metrics"]["test_accuracy"] >= 0.9


# ---------- API ----------

@pytest.fixture
def client(model_dir, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "MODEL_DIR", model_dir)
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_setosa(client):
    r = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5,
                                      "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 200
    body = r.json()
    assert body["species"] == "setosa"
    assert 0 <= body["confidence"] <= 1


def test_predict_invalid_input(client):
    r = client.post("/predict", json={"sepal_length": "abc"})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json={"sepal_length": 6.7, "sepal_width": 3.0,
                                  "petal_length": 5.2, "petal_width": 2.3})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "iris_predictions_total" in r.text
