import json
from pathlib import Path

import joblib
import pytest
from tensorflow import keras

from pipeline.ast_tokenizer import tokenize_source
from pipeline.winnowing import winnow
from pipeline.features import compute_features, FEATURE_COLUMNS

MODEL_PATH = Path("plagiarism_model.keras")
SCALER_PATH = Path("scaler.joblib")
CONFIG_PATH = Path("decision_threshold.json")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = json.load(f)

MODEL = keras.models.load_model(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)


def predict_pair(code_a: str, code_b: str):
    ta = tokenize_source(code_a, masking=CFG["masking"])
    tb = tokenize_source(code_b, masking=CFG["masking"])
    if ta["error"]:
        raise ValueError(ta["error"])
    if tb["error"]:
        raise ValueError(tb["error"])

    fps_a = winnow(ta["tokens"], k=int(CFG["k"]), w=int(CFG["w"]))
    fps_b = winnow(tb["tokens"], k=int(CFG["k"]), w=int(CFG["w"]))
    feats = compute_features(
        ta["tokens"], ta["max_depth"], fps_a,
        tb["tokens"], tb["max_depth"], fps_b,
        k=int(CFG["k"]),
    )
    x_new = SCALER.transform([[feats[name] for name in FEATURE_COLUMNS]])
    prob = float(MODEL.predict(x_new, verbose=0)[0][0])
    return prob


def test_tokenize_source_accepts_valid_python():
    result = tokenize_source("def f(x):\n    return x + 1\n")
    assert result["error"] is None
    assert result["tokens"]


def test_predict_pair_returns_probability_between_0_and_1():
    prob = predict_pair(
        "def suma(x, y):\n    return x + y\n",
        "def add(a, b):\n    return a + b\n",
    )
    assert 0.0 <= prob <= 1.0


def test_predict_pair_rejects_invalid_python():
    with pytest.raises(ValueError):
        predict_pair("def f(:\n    pass\n", "def g():\n    return 1\n")
