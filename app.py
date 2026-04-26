import streamlit as st
import requests
import pandas as pd
import joblib
import time
import os
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Smart Room Monitoring",
    page_icon="🏠",
    layout="wide"
)

# ===============================
# ATTRACTIVE BACKGROUND & FONT COLORS
# ===============================
st.markdown("""
<style>
/* Background Gradient */
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: #ffffff;
}

/* Headers */
h1, h2, h3 {
    color: #ffffff;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #ffffff, #d0d4e0);
    color: #222222 !important;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0px 18px 40px rgba(0,0,0,0.35);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50, #4ca1af);
    color: white;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#ff7e5f,#feb47b);
    color: white;
    font-weight: bold;
    border-radius:12px;
    padding: 8px 24px;
    transition: transform 0.3s;
}
.stButton>button:hover{
    transform: scale(1.05);
}

/* Graph containers */
div[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 8px;
}

/* Table */
div[data-testid="stDataFrameContainer"] {
    background: rgba(255, 255, 255, 0.1);
    color: white;
    border-radius: 12px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.title("🏠 Smart Room Condition Monitoring Dashboard")
st.caption("📊 Attractive UI • Live IoT Sensors • ML Prediction • Gauges & Meters")

# ===============================
# LOAD ML MODELS
# ===============================
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return rf_model, scaler, label_encoder

rf_model, scaler, label_encoder = load_models()

# ===============================
# BLYNK CONFIGURATION
# ===============================
BLYNK_TOKEN = "uyyj0idr0hKbeadefrT6vZnBuCZCxg_7"
URLS = {
    "Temperature": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V0",
    "Humidity": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V1",
    "Moisture": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V2",
    "MQ135": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V3",
    "AC_Voltage": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V4",
    "AC_Current": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V5",
    "Dust": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V6",
}
UPDATE_RESULT_V9 = f"https://blynk.cloud/external/api/update?token={BLYNK_TOKEN}&V9="

# ===============================
# FUNCTIONS
# ===============================
def fetch(url):
    return float(requests.get(url, timeout=5).text)

def get_sensor_data():
    return {k: fetch(v) for k, v in URLS.items()}

def predict_condition(data):
    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    pred = rf_model.predict(scaled)
    confidence = rf_model.predict_proba(scaled).max()
    return label_encoder.inverse_transform(pred)[0], confidence

# ===============================
# CSV LOGGING
# ===============================
CSV_FILE = "room_history.csv"
def log_to_csv(data, result, confidence):
    row = data.copy()
    row["Condition"] = result
    row["Confidence"] = round(confidence*100,2)
    row["Timestamp"] = datetime.now()
    df = pd.DataFrame([row])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_FILE, index=False)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙ Controls")
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (10s)")
show_graphs = st.sidebar.checkbox("📈 Show Graphs", True)
enable_logging = st.sidebar.checkbox("💾 CSV Logging", True)

# ===============================
# MAIN DASHBOARD
# ===============================
try:
    sensor_data = get_sensor_data()
    result, confidence = predict_condition(sensor_data)

    # ---- LIVE METRICS ----
    st.subheader("📊 Live Sensor Metrics")
    cols = st.columns(4)
    for i, (k, v) in enumerate(sensor_data.items()):
        cols[i%4].metric(label=k, value=round(v,2))

    # ---- ROOM CONDITION ----
    st.subheader("🧠 Room Condition")
    if result == "Excellent":
        st.success(f"🟢 {result}")
    elif result == "Good":
        st.info(f"🔵 {result}")
    elif result == "Moderate":
        st.warning(f"🟡 {result}")
    elif result == "Poor":
        st.error(f"🔴 {result}")

    # Confidence bar
    st.progress(int(confidence*100))

    # Send to Blynk
    requests.get(UPDATE_RESULT_V9 + result)

    # CSV Logging
    if enable_logging:
        log_to_csv(sensor_data, result, confidence)

    # ---- GAUGES & METERS ----
    st.subheader("🌡 Sensor Gauges & Meters")
    gauge_col1, gauge_col2 = st.columns(2)

    # Temperature Gauge
    fig_temp = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sensor_data["Temperature"],
        title={'text': "Temperature (°C)"},
        gauge={'axis': {'range':[0,50]}, 'bar': {'color':'red'}}
    ))
    gauge_col1.plotly_chart(fig_temp, use_container_width=True)

    # Humidity Gauge
    fig_hum = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sensor_data["Humidity"],
        title={'text': "Humidity (%)"},
        gauge={'axis': {'range':[0,100]}, 'bar': {'color':'blue'}}
    ))
    gauge_col2.plotly_chart(fig_hum, use_container_width=True)

    # ---- HISTORY & DOWNLOAD ----
    if show_graphs:
        st.subheader("📈 Sensor History")
        try:
            history = pd.read_csv(CSV_FILE, parse_dates=["Timestamp"], on_bad_lines='skip').tail(50)
            sensor = st.selectbox("Select Sensor", URLS.keys())
            fig, ax = plt.subplots()
            ax.plot(history["Timestamp"], history[sensor])
            ax.set_title(f"{sensor} Trend")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("📋 History Data")
            st.dataframe(history, use_container_width=True)

            st.download_button(
                "⬇ Download History CSV",
                data=open(CSV_FILE, "rb"),
                file_name="room_condition_history.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.warning(f"⚠ Could not read CSV properly: {e}")

except Exception as e:
    st.error(f"❌ Error fetching or processing data: {e}")

# ===============================
# AUTO REFRESH
# ===============================
if auto_refresh:
    time.sleep(10)
    st.rerun()
