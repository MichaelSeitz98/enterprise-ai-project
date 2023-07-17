from fastapi import FastAPI
from pydantic import BaseModel
from ml_model import *
import json
import yaml

app = FastAPI()

class PredictionRequest(BaseModel):
    LivingSpace: int
    Rooms: int
    altbau_(bis_1945): bool # type: ignore
    balkon: bool
    barriefrei: bool
    dachgeschoss: bool
    einbaukueche: bool
    neubau: bool
    parkett: bool
    stellplatz: bool
    bad_wc_getrennt: bool
    personenaufzug: bool
    garten: bool
    garage: bool
    renoviert: bool
    terrasse: bool
    wanne: bool
    zentralheizung: bool
    abstellraum: bool
    ferne: bool
    fussbodenheizung: bool
    gartennutzung: bool
    kelleranteil: bool
    ZipCode: str

@app.post("/predict")
async def predict_housing_price(data: dict):
    result = predict(data)
    return {"Predicted price": result}

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

