from fastapi.testclient import TestClient
from app import app
import numpy as np
import pandas as pd
import joblib

#Models
lr = joblib.load("gas_model.pkl") # Load "model.pkl"
model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"

client = TestClient(app)
def test_main_resource():
    response_auth = client.get("/")
    assert response_auth.status_code == 200



