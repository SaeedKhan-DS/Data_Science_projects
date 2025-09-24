# Austrian Energy Consumption Prediction and Optimization using Machine Learning

## ⚡ Project Overview
This project focuses on predicting energy consumption in Austria and identifying optimization strategies using machine learning. The system combines historical energy load data with weather information to forecast demand and simulate potential energy-saving techniques.

## 🎯 Key Objectives
- Predict hourly energy load consumption for Austria
- Identify key factors influencing energy demand
- Simulate optimization strategies for energy savings
- Deploy the model as an interactive web application

## 📊 Dataset
- **Energy Data Source**: Open Power System Data (OPSD)
- **Weather Data Source**: Open-Meteo API
- **Time Period**: January 2015 - January 2020 (5 years)
- **Records**: 43,848 hourly observations
- **Features**: 24 columns including energy load, weather variables, and time-based features

## 🛠️ Technical Implementation

### Data Preprocessing
- Handled missing values in `AT_price_day_ahead`, `AT_solar_generation_actual`, and `AT_wind_onshore_generation_actual`
- Applied linear interpolation for time series data integrity
- Engineered comprehensive features for improved prediction

### Feature Engineering
- **Time-based Features**: Hour of day, day of week, weekend indicator, month, Austrian holidays
- **Lagged Features**: Energy load from previous 24, 48, and 72 hours
- **Weather Features**: Temperature, dew point, and other meteorological variables

### Machine Learning Model
- **Algorithm**: LightGBM Regressor
- **Preprocessing**: StandardScaler for feature normalization
- **Training Approach**: Chronological train-test split with early stopping
- **Performance Metrics**:
  - R-squared (R²): 0.96
  - RMSE: 272.43
  - MSE: 74,216.33

## 📈 Key Insights from EDA
- Strong annual and weekly seasonality patterns
- Distinct load profiles for weekdays vs. weekends
- Significant inverse correlation between temperature/dew point and energy load
- Morning and evening peak consumption during weekdays

## 💡 Optimization Strategies & Results

### Temperature-Based Optimization
- Adjust thermostat setpoints based on temperature ranges
- **Potential Savings**: 2,852,922.64 MWh

### Weekday Peak Hour Optimization
- Reduce load during identified peak hours (8-10 AM, 6-8 PM)
- **Potential Savings**: 4,827,181.25 MWh

## 🌐 Deployment

### FastAPI Backend
- RESTful API for model predictions
- Endpoints for energy load forecasting and optimization simulations
- Real-time data processing capabilities

### Streamlit Frontend
- Interactive web interface for data visualization
- User-friendly input forms for custom scenarios
- Real-time prediction results and savings estimates

### API Endpoints
- `POST /predict` - Get energy load predictions
- `POST /optimize` - Simulate optimization scenarios
- `GET /health` - API status check
