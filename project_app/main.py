import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Optional
import joblib
from sklearn.preprocessing import StandardScaler
import holidays
import json

# Initialize FastAPI app
app = FastAPI(title="Austrian Energy Consumption Prediction API", 
              description="API for predicting energy consumption and optimizing energy usage")

# Custom JSON encoder to handle datetime and pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super().default(obj)

# Set the custom JSON encoder for the FastAPI app
app.json_encoder = CustomJSONEncoder

# Load your trained model 
try:
    model = joblib.load('lgbm_energy_prediction_model.pkl')
except:
    # Create a placeholder model if loading fails
    print("Model file not found, using placeholder model")
    model = None

# Austrian holiday calendar 
class AustrianHolidayCalendar(holidays.HolidayBase):
    def __init__(self, **kwargs):
        self.country = 'AT'
        super().__init__(**kwargs)

# Feature engineering functions 
def create_features(dt):
    """Create time-based features from datetime"""
    return {
        "hour": dt.hour,
        "week_day": dt.weekday(),
        "weekend": 1 if dt.weekday() >= 5 else 0,
        "month": dt.month
    }

def is_austrian_holiday(dt):
    """Check if date is an Austrian holiday"""
    cal = AustrianHolidayCalendar()
    return 1 if dt in cal else 0

# Input data model
class EnergyData(BaseModel):
    datetime: str
    AT_load_forecast_entsoe_transparency: float
    AT_price_day_ahead: float
    AT_solar_generation_actual: Optional[float] = None
    AT_wind_onshore_generation_actual: Optional[float] = None
    temperature_2m: float
    relative_humidity_2m: int
    dew_point_2m: float
    apparent_temperature: float
    precipitation: float
    rain: float
    snowfall: float
    cloud_cover: int
    cloud_cover_low: int
    cloud_cover_mid: int
    cloud_cover_high: int
    wind_speed_10m: float
    wind_direction_10m: int
    wind_gusts_10m: float
    surface_pressure: float
    shortwave_radiation: float
    direct_radiation: float
    diffuse_radiation: float
    load_lag_24: Optional[float] = None
    load_lag_48: Optional[float] = None
    load_lag_72: Optional[float] = None

class PredictionRequest(BaseModel):
    data: List[EnergyData]

class PredictionResponse(BaseModel):
    predictions: List[float]
    timestamps: List[str]

class OptimizationRequest(BaseModel):
    budget: float
    time_range: str  # "day", "week", "month"
    max_consumption: Optional[float] = None

class OptimizationResponse(BaseModel):
    optimal_schedule: List[dict]
    estimated_cost: float
    estimated_savings: float
    recommended_actions: List[str]

# Model prediction function
def predict_energy_consumption(features_df):
    """
    Predict energy consumption using the trained model or fallback
    """
    if model is not None:
        # Use the actual trained model
        return model.predict(features_df)
    else:
        # Fallback mock prediction
        print("Using fallback prediction model")
        base_load = 7000
        temp_effect = -50 * features_df['temperature_2m']  # Higher temp = lower consumption
        hour_effect = 200 * np.sin(2 * np.pi * features_df['hour'] / 24)
        
        return base_load + temp_effect + hour_effect + np.random.normal(0, 100, len(features_df))

@app.post("/predict", response_model=PredictionResponse)
async def predict_consumption(request: PredictionRequest):
    """
    Predict energy consumption for given input data
    """
    try:
        data_list = []
        timestamps = []
        
        for item in request.data:
            # Convert to datetime
            dt = datetime.fromisoformat(item.datetime.replace('Z', '+00:00'))
            
            # Create feature dictionary
            features = {
                "AT_load_forecast_entsoe_transparency": item.AT_load_forecast_entsoe_transparency,
                "AT_price_day_ahead": item.AT_price_day_ahead,
                "temperature_2m": item.temperature_2m,
                "relative_humidity_2m": item.relative_humidity_2m,
                "dew_point_2m": item.dew_point_2m,
                "apparent_temperature": item.apparent_temperature,
                "precipitation": item.precipitation,
                "rain": item.rain,
                "snowfall": item.snowfall,
                "cloud_cover": item.cloud_cover,
                "cloud_cover_low": item.cloud_cover_low,
                "cloud_cover_mid": item.cloud_cover_mid,
                "cloud_cover_high": item.cloud_cover_high,
                "wind_speed_10m": item.wind_speed_10m,
                "wind_direction_10m": item.wind_direction_10m,
                "wind_gusts_10m": item.wind_gusts_10m,
                "surface_pressure": item.surface_pressure,
                "shortwave_radiation": item.shortwave_radiation,
                "direct_radiation": item.direct_radiation,
                "diffuse_radiation": item.diffuse_radiation,
                "load_lag_24": item.load_lag_24 if item.load_lag_24 else 0,
                "load_lag_48": item.load_lag_48 if item.load_lag_48 else 0,
                "load_lag_72": item.load_lag_72 if item.load_lag_72 else 0,
            }
            
            # Add time-based features
            time_features = create_features(dt)
            features.update(time_features)
            features["holiday"] = is_austrian_holiday(dt.date())
            
            # Handle optional fields
            if item.AT_solar_generation_actual is not None:
                features["AT_solar_generation_actual"] = item.AT_solar_generation_actual
            else:
                features["AT_solar_generation_actual"] = 0
                
            if item.AT_wind_onshore_generation_actual is not None:
                features["AT_wind_onshore_generation_actual"] = item.AT_wind_onshore_generation_actual
            else:
                features["AT_wind_onshore_generation_actual"] = 0
            
            data_list.append(features)
            timestamps.append(item.datetime)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(data_list)
        
        # Make predictions
        predictions = predict_energy_consumption(features_df)
        
        # Use jsonable_encoder to ensure proper serialization
        response_data = {
            "predictions": predictions.tolist(),
            "timestamps": timestamps
        }
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_energy_usage(request: OptimizationRequest):
    """
    Optimize energy usage based on predictions and constraints
    """
    try:
        # This is a simplified optimization example
        if request.time_range == "day":
            hours = 24
        elif request.time_range == "week":
            hours = 168
        else:  # month
            hours = 720
        
        # Generate mock optimal schedule
        optimal_schedule = []
        for i in range(hours):
            hour = i % 24
            # Simple logic: shift consumption to cheaper hours (night)
            recommended_load = 6000 if hour < 6 or hour > 22 else 8000
            
            if request.max_consumption and recommended_load > request.max_consumption:
                recommended_load = request.max_consumption
                
            optimal_schedule.append({
                "hour": hour,
                "recommended_consumption": recommended_load,
                "estimated_cost": recommended_load * 0.035  # Mock price
            })
        
        # Calculate totals
        total_cost = sum(item["estimated_cost"] for item in optimal_schedule)
        estimated_savings = request.budget - total_cost if request.budget else total_cost * 0.15
        
        # Generate recommendations
        recommendations = [
            "Shift high-consumption activities to off-peak hours (10 PM - 6 AM)",
            "Consider using solar generation during peak sunlight hours",
            "Monitor weather forecasts for optimal renewable energy usage"
        ]
        
        # Use jsonable_encoder for proper serialization
        response_data = {
            "optimal_schedule": optimal_schedule,
            "estimated_cost": total_cost,
            "estimated_savings": max(0, estimated_savings),
            "recommended_actions": recommendations
        }
        
        return OptimizationResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Austrian Energy Consumption Prediction API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)