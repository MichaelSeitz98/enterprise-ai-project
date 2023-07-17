from fastapi import FastAPI
from pydantic import BaseModel
from predict_request_base_models import *
from ml_model import *
import json
import yaml

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse, summary="Predict housing price")
async def predict_housing_price(data: PredictionRequest):
    result = predict(data)
    return {"Predicted_price": result}

@app.get("/model-info")
async def model_info():
    with open("prod_model_details.txt", "r") as file:
        content = file.read()
    
    # Parse YAML content and convert it to JSON
    json_content = yaml.safe_load(content)
    
    return json_content

@app.get("/")
async def root():
    return {"Status": "Health check - API is running"}

