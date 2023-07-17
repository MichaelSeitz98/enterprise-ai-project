from pydantic import BaseModel

class PredictionRequest(BaseModel):
    LivingSpace: float
    Rooms: float
    altbau_bis_1945: bool 
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

class PredictionResponse(BaseModel):
    Predicted_price: float