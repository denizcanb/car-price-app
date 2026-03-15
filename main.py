from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import os

app = FastAPI(title="Car Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "car_price_pipe.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model yüklendi: {MODEL_PATH}")
    except Exception as e:
        print(f"Model yüklenemedi: {e}")

class CarInput(BaseModel):
    Mileage:  float = Field(..., example=15000)
    Make:     str   = Field(..., example="Buick")
    Model:    str   = Field(..., example="Century")
    Trim:     str   = Field(..., example="Sedan 4D")
    Type:     str   = Field(..., example="Sedan")
    Cylinder: int   = Field(..., example=6)
    Liter:    float = Field(..., example=3.1)
    Doors:    int   = Field(..., example=4)
    Cruise:   int   = Field(..., example=1)
    Sound:    int   = Field(..., example=1)
    Leather:  int   = Field(..., example=0)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(car: CarInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    input_df = pd.DataFrame([{
        "Mileage": car.Mileage, "Make": car.Make, "Model": car.Model,
        "Trim": car.Trim, "Type": car.Type, "Cylinder": car.Cylinder,
        "Liter": car.Liter, "Doors": car.Doors, "Cruise": car.Cruise,
        "Sound": car.Sound, "Leather": car.Leather,
    }])
    try:
        price = float(model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")
    return {"predicted_price": round(price, 2), "formatted_price": f"${price:,.0f}", "currency": "USD"}

@app.get("/options")
def get_options():
    return {
        "makes": ["Buick", "Cadillac", "Chevrolet", "Pontiac", "SAAB", "Saturn"],
        "models": {
            "Buick": ["Century", "LeSabre", "Park Avenue", "Rainier", "Rendezvous", "Terraza"],
            "Cadillac": ["CTS", "DeVille", "Escalade ESV", "Escalade EXT", "Seville"],
            "Chevrolet": ["Aveo", "Blazer", "Cavalier", "Classic", "Colorado", "Corvette",
                          "Impala", "Malibu", "Monte Carlo", "Silverado", "Suburban",
                          "Tahoe", "TrailBlazer", "Uplander", "Venture"],
            "Pontiac": ["Aztek", "Bonneville", "Grand Am", "Grand Prix", "GTO",
                        "Montana SV6", "Pursuit", "Sunfire", "Vibe"],
            "SAAB": ["9-2X", "9-3", "9-5"],
            "Saturn": ["Ion", "L Series", "Relay", "VUE"],
        },
        "types": ["Sedan", "Coupe", "Hatchback", "Convertible", "Wagon", "SUV", "Truck", "Van"],
        "cylinders": [4, 6, 8],
        "doors": [2, 4],
    }
