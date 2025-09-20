import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# CONFIG
# ==============================
FASTAPI_URL = "http://127.0.0.1:8000"  
DATA_PATH = "clean_energy_weather_data.csv"

st.set_page_config(
    page_title="⚡ Austrian Energy Prediction",
    page_icon="⚡",
    layout="wide"
)

# ==============================
# STYLING
# ==============================
st.markdown(
    """
    <style>
    .main {background-color: #f8fafc;}
    .stButton>button {
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e40af, #0891b2);
        transform: scale(1.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    return df

data = load_data()

# ==============================
# SIDEBAR NAV
# ==============================
st.sidebar.title("⚡ Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["📊 Data Overview", "🔮 Predict Consumption", "🛠 Optimize Usage", "📈 API Health"]
)

# ==============================
# PAGE 1: DATA OVERVIEW
# ==============================
if page == "📊 Data Overview":
    st.title("📊 Energy Consumption prediction and optimization")
    st.caption("Explore the dataset powering the prediction model.")

    # Top KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(data):,}")
    col2.metric("Date Range", f"{data['datetime'].min().date()} → {data['datetime'].max().date()}")
    col3.metric("Avg Load Forecast", f"{data['AT_load_forecast_entsoe_transparency'].mean():.2f} MW")

    st.divider()
    st.subheader("Time Series: Forecasted Load")
    fig = px.line(
        data,
        x="datetime",
        y="AT_load_forecast_entsoe_transparency",
        title="Forecasted Energy Load Over Time",
        labels={"AT_load_forecast_entsoe_transparency": "Load (MW)"},
        template="plotly_white"
    )
    fig.update_traces(line=dict(color="#2563eb"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Correlation Heatmap")
    corr = data.corr(numeric_only=True)
    heatmap = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="Tealrose",
        zmin=-1, zmax=1
    ))
    st.plotly_chart(heatmap, use_container_width=True)

# ==============================
# PAGE 2: PREDICT CONSUMPTION
# ==============================
elif page == "🔮 Predict Consumption":
    st.title("🔮 Predict Energy Consumption")
    st.caption("Enter conditions to predict Austrian energy usage.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            dt = st.text_input("Datetime (ISO)", value=datetime.utcnow().isoformat()+"Z")
            temp = st.number_input("Temperature (°C)", value=15.0)
            humidity = st.number_input("Relative Humidity (%)", value=50)
            load_forecast = st.number_input("Load Forecast (MW)", value=7000.0)
        with col2:
            price = st.number_input("Day Ahead Price", value=100.0)
            wind = st.number_input("Wind Onshore Generation (MW)", value=500.0)
            solar = st.number_input("Solar Generation (MW)", value=200.0)

        submitted = st.form_submit_button("🚀 Predict Now")

    if submitted:
        payload = {
            "data": [{
                "datetime": dt,
                "AT_load_forecast_entsoe_transparency": load_forecast,
                "AT_price_day_ahead": price,
                "AT_solar_generation_actual": solar,
                "AT_wind_onshore_generation_actual": wind,
                "temperature_2m": temp,
                "relative_humidity_2m": humidity,
                "dew_point_2m": 10,
                "apparent_temperature": temp,
                "precipitation": 0,
                "rain": 0,
                "snowfall": 0,
                "cloud_cover": 20,
                "cloud_cover_low": 10,
                "cloud_cover_mid": 10,
                "cloud_cover_high": 10,
                "wind_speed_10m": 3.5,
                "wind_direction_10m": 180,
                "wind_gusts_10m": 5,
                "surface_pressure": 1010,
                "shortwave_radiation": 100,
                "direct_radiation": 50,
                "diffuse_radiation": 30
            }]
        }

        try:
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prediction Successful!")

                # Metrics
                pred_val = result["predictions"][0]
                col1, col2 = st.columns(2)
                col1.metric("Predicted Load (MW)", f"{pred_val:.2f}")
                col2.metric("Timestamp", result["timestamps"][0])

                # Chart
                fig = px.bar(
                    x=result["timestamps"],
                    y=result["predictions"],
                    labels={"x": "Timestamp", "y": "Predicted Load (MW)"},
                    title="Predicted Energy Consumption",
                    template="plotly_white"
                )
                fig.update_traces(marker_color="#06b6d4")
                st.plotly_chart(fig, use_container_width=True)

                # Download button
                df = pd.DataFrame(result)
                st.download_button("⬇️ Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")

            else:
                st.error(f"❌ API Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# ==============================
# PAGE 3: OPTIMIZE USAGE
# ==============================
elif page == "🛠 Optimize Usage":
    st.title("🛠 Optimize Energy Usage")
    st.caption("Find the most efficient usage schedule under constraints.")

    with st.form("opt_form"):
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Budget (€)", value=50000.0)
            time_range = st.selectbox("Time Range", ["day", "week", "month"])
        with col2:
            max_consumption = st.number_input("Max Consumption (MW)", value=8000.0)

        submitted = st.form_submit_button("⚡ Optimize")

    if submitted:
        payload = {
            "budget": budget,
            "time_range": time_range,
            "max_consumption": max_consumption
        }
        try:
            response = requests.post(f"{FASTAPI_URL}/optimize", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Optimization Completed!")

                # Metrics
                col1, col2 = st.columns(2)
                col1.metric("Estimated Cost (€)", f"{result['estimated_cost']:.2f}")
                col2.metric("Estimated Savings (€)", f"{result['estimated_savings']:.2f}")

                # Schedule chart
                schedule_df = pd.DataFrame(result["optimal_schedule"])
                fig = px.line(
                    schedule_df,
                    x="hour",
                    y="recommended_consumption",
                    title="Optimal Consumption Schedule",
                    labels={"recommended_consumption": "Consumption (MW)"},
                    template="plotly_white"
                )
                fig.update_traces(line_color="#2563eb")
                st.plotly_chart(fig, use_container_width=True)

                # Recommendations
                st.subheader("📌 Recommendations")
                for action in result["recommended_actions"]:
                    st.markdown(f"- ✅ {action}")

                # Download schedule
                st.download_button("⬇️ Download Schedule", schedule_df.to_csv(index=False), "schedule.csv", "text/csv")

            else:
                st.error(f"❌ API Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# ==============================
# PAGE 4: API HEALTH
# ==============================
elif page == "📈 API Health":
    st.title("📈 API Health Check")

    try:
        response = requests.get(f"{FASTAPI_URL}/health")
        if response.status_code == 200:
            st.success("✅ API is Healthy")
            st.json(response.json())
        else:
            st.error(f"❌ API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")
