# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
import os

# setting page configuration
st.set_page_config(layout="wide", page_title="Wind Power Forecast", page_icon="⚡")

# suppressing oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# loading model and data with caching
@st.cache_resource
def load_system():
    model = load_model('./model_assets/wind_lstm_model.keras')
    scaler = joblib.load('./model_assets/scaler.gz')
    features = joblib.load('./model_assets/feature_cols.pkl')
    raw_data = pd.read_csv('./model_assets/test_data.csv')
    return model, scaler, features, raw_data


model, scaler, feature_cols, df_test = load_system()

# setting up the sidebar
st.sidebar.title("Wind Turbine Control Station⚡")
st.sidebar.subheader("Site: Location 1 - Onshore")

# setting up the time selection slider
max_time = len(df_test) - 50
valid_indices = list(range(30, max_time))

# setting up a function to format hour display
def format_hour(hour_idx):
    day = (hour_idx // 24) + 1
    hour = hour_idx % 24
    return f"Day {day} | {hour:02d}:00"

# setting up the simulation time slider
sim_time = st.sidebar.select_slider(
    "**Scroll to simulate different operational days**",
    options=valid_indices,
    value=100,
    format_func=format_hour,
)

# setting up the lookahead horizon slider
horizon = st.sidebar.slider("**Adjust Lookahead Hours**", 1, 24, 12)

# setting up the ground truth checkbox
show_truth = st.sidebar.checkbox("Show Ground Truth (Validation Mode)", value=False)

# setting up the system state display
current_day = (sim_time // 24) + 1
current_hour = sim_time % 24
st.sidebar.markdown("---")
st.sidebar.info(f"**System State:**\n\n🕒 **Day:** {current_day}\n\n⏱️ **Time:** {current_hour:02d}:00")


# processing data
past_window = df_test.iloc[sim_time-24 : sim_time]

# running predictions
pred_values = []
actual_values = []
time_steps = []
LOOKBACK = 24

for h in range(horizon):
    current_step = sim_time + h
    
    raw_window = df_test[feature_cols].iloc[current_step-LOOKBACK : current_step]
    
    scaled_window = scaler.transform(raw_window)
    

    input_tensor = scaled_window.reshape(1, LOOKBACK, len(feature_cols))

    prediction = model.predict(input_tensor, verbose=0)[0][0]
    
    pred_values.append(prediction)
    actual_values.append(df_test['Power'].iloc[current_step])
    time_steps.append(f"T+{h+1}")

# building the Streamlit layout
st.markdown("<h2 style='text-align: center; color: #555555;'>⚙️ Wind Power Forecast Simulation & Grid Control Interface</h2>", unsafe_allow_html=True)

# setting up two columns for graphs
col_left, col_right = st.columns(2)

# setting the left graph: operational data
with col_left:
    st.subheader("📡 Operational Data (Past 24 hours)")
    
    fig_left = go.Figure()
    
    # plotting wind speed
    fig_left.add_trace(go.Scatter(
        x=list(range(-24, 0)), 
        y=past_window['windspeed_100m'],
        name="Wind Speed (m/s)",
        line=dict(color='#0068C9', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 104, 201, 0.1)'
    ))
    
    # plotting temperature
    fig_left.add_trace(go.Scatter(
        x=list(range(-24, 0)), 
        y=past_window['temperature_2m'],
        name="Temp (°C)",
        line=dict(color='#FF2B2B', width=1, dash='dot'),
        yaxis="y2"
    ))

    fig_left.update_layout(
        xaxis_title="Time Relative to Now (Hours)",
        yaxis_title="Wind Speed (m/s)",
        yaxis2=dict(title="Temp (°C)", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig_left, width='stretch')
    
    # setting up currrent weaather conditions
    st.markdown("#### 🌤️ Current Weather Conditions")
    c1, c2 = st.columns(2)
    c1.metric("Current Wind Speed", f"{past_window['windspeed_100m'].iloc[-1]:.1f} m/s")
    c2.metric("Current Temperature", f"{past_window['temperature_2m'].iloc[-1]:.1f} °C")


# setting the right graph: power forecast
with col_right:
    st.subheader(f"🔌Power Forecast (Next {horizon} Hours)")
    
    fig_right = go.Figure()
    
    # plotting LSTM Forecast
    fig_right.add_trace(go.Scatter(
        x=time_steps, 
        y=pred_values,
        name="LSTM Forecast",
        line=dict(color='#00CC96', width=3),
        mode='lines+markers'
    ))
    
    # plotting ground truth if enabled
    if show_truth:
        fig_right.add_trace(go.Scatter(
            x=time_steps, 
            y=actual_values,
            name="Actual (Sim)",
            line=dict(color='grey', width=1, dash='dot')
        ))

    fig_right.update_layout(
        xaxis_title="Future Horizon",
        yaxis=dict(title="Normalized Power (0-1)", range=[0, 1.1]),
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig_right, width='stretch')

    # setting grid dispatch insights
    avg_power = np.mean(pred_values)
    volatility = np.std(pred_values)
    
    st.markdown("#### ℹ️ Grid Dispatch Insights")
    
    if volatility > 0.15:
        st.error(f"**Action:** ⚠️ **Ramp Alert.** High volatility detected (Standard Deviation: {volatility:.2f}). Pre-charge battery storage.")
    elif avg_power > 0.8:
        st.success(f"**Action:** 🟢 **Peak Generation.** Forecast average {avg_power:.0%} capacity. Export excess to main grid.")
    else:
        st.info(f"**Action:** 🔵 **Steady State.** Grid stable. Maintain standard dispatch protocols.")
