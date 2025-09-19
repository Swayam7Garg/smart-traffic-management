import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import glob
import holidays

# --- Load the trained model ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")

MODEL_NAME = None
if os.path.exists(MODEL_DIR):
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith("_traffic_model.pkl"):
            MODEL_NAME = fname
            break

if MODEL_NAME is None:
    st.error("No trained model found in models folder.")
    st.stop()

model = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))

## --- Load actual data for visualization ---
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "traffic.csv")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
else:
    df = None

# --- Junction options (update with actual names if needed) ---
if df is not None:
    junction_options = sorted(df['Junction'].unique())
else:
    junction_options = []
junction_mapping = {name: idx for idx, name in enumerate(junction_options)}

# --- Sidebar UI ---
# st.sidebar.title("Traffic Prediction Input")
# date = st.sidebar.date_input("Select Date", value=datetime.today())
# time = st.sidebar.time_input("Select Time", value=datetime.now().time())
# junction = st.sidebar.selectbox("Select Junction", options=junction_options)

# st.title("🚦 AI-Powered Traffic Prediction System")
# st.write("Predict vehicle count for a given date, time, and junction. Visualize traffic trends and compare predictions with actual data.")

# if st.sidebar.button("Predict"):
#     dt = datetime.combine(date, time)
#     year = dt.year
#     month = dt.month
#     day = dt.day
#     hour = dt.hour
#     weekday = dt.weekday()
#     hour_sin = np.sin(2 * np.pi * hour / 24)
#     hour_cos = np.cos(2 * np.pi * hour / 24)
#     weekday_sin = np.sin(2 * np.pi * weekday / 7)
#     weekday_cos = np.cos(2 * np.pi * weekday / 7)
#     junction_enc = junction_mapping[junction]

#     input_df = pd.DataFrame([{
#         'Junction_enc': junction_enc,
#         'Year': year,
#         'Month': month,
#         'Day': day,
#         'Hour': hour,
#         'Weekday': weekday,
#         'Hour_sin': hour_sin,
#         'Hour_cos': hour_cos,
#         'Weekday_sin': weekday_sin,
#         'Weekday_cos': weekday_cos
#     }])

#     pred = model.predict(input_df)[0]
#     st.success(f"### Predicted Vehicle Count: {int(pred)}")
    # ...existing sidebar UI...
st.sidebar.title("Traffic Prediction Input")
date = st.sidebar.date_input("Select Date", value=datetime.today())
indian_holidays = holidays.country_holidays('IN', years=[date.year])
is_holiday = date in indian_holidays

# Get next 10 holidays/festivals in the selected year
upcoming = [(d, name) for d, name in sorted(indian_holidays.items()) if d >= date][:10]
if upcoming:
    st.sidebar.markdown("**Upcoming Festivals/Holidays:**")
    for d, name in upcoming:
        symbol = "🛕" if d == date else "•"
        st.sidebar.markdown(f"{symbol} {d.strftime('%b %d')}: {name}")
time = st.sidebar.time_input("Select Time", value=datetime.now().time())
junction = st.sidebar.selectbox("Select Junction", options=junction_options)
special_event = st.sidebar.checkbox("Festival/Holiday? (Special Event)", value=False)

# ...inside the prediction block...
if st.sidebar.button("Predict"):
    dt = datetime.combine(date, time)
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    weekday = dt.weekday()
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    junction_enc = junction_mapping[junction]

    # --- Hypothetical traffic adjustment for special events ---
    input_df = pd.DataFrame([{
        'Junction_enc': junction_enc,
        'Year': year,
        'Month': month,
        'Day': day,
        'Hour': hour,
        'Weekday': weekday,
        'Hour_sin': hour_sin,
        'Hour_cos': hour_cos,
        'Weekday_sin': weekday_sin,
        'Weekday_cos': weekday_cos
    }])

    pred = model.predict(input_df)[0]

    # Apply adjustment for special event or detected holiday
    if special_event or is_holiday:
        pred = pred * 1.3
        st.info("Special event or holiday detected: Traffic prediction increased by 30%.")

    st.success(f"### Predicted Vehicle Count: {int(pred)}")

    # ...rest of your code (alerts, suggestions, visualizations)...
    # --- Peak-hour indicators ---
    if df is not None:
        actual = df[(df['Junction'] == junction) & (df['DateTime'].dt.date == date)]
        if not actual.empty:
            actual_hourly = actual.groupby(actual['DateTime'].dt.hour)['Vehicles'].mean()
            top_hours = actual_hourly.sort_values(ascending=False).head(3)
            st.info(f"🔝 Peak hours for {junction} on {date}: " +
                    ", ".join([f"{int(h)}:00 ({int(cnt)} vehicles)" for h, cnt in top_hours.items()]))
            

    congestion_threshold = 80  # Set this based on your data distribution
    if pred > congestion_threshold:
        st.warning(f"🚦 High congestion expected at {junction} around {hour}:00. Consider alternate routes or travel times.")

        # --- Travel Suggestions ---
        # Suggest alternative hours with lower predicted traffic
        alt_hours = []
        for h in range(24):
            if h == hour:
                continue
            hour_sin = np.sin(2 * np.pi * h / 24)
            hour_cos = np.cos(2 * np.pi * h / 24)
            input_df_alt = pd.DataFrame([{
                'Junction_enc': junction_enc,
                'Year': year,
                'Month': month,
                'Day': day,
                'Hour': h,
                'Weekday': weekday,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Weekday_sin': weekday_sin,
                'Weekday_cos': weekday_cos
            }])
            alt_pred = model.predict(input_df_alt)[0]
            if alt_pred < congestion_threshold:
                alt_hours.append(h)
        if alt_hours:
            st.info(f"Suggested alternative travel times at {junction} with lower traffic: {', '.join([f'{h}:00' for h in alt_hours])}")
        else:
            st.info("No alternative travel times found with lower traffic for this day.")
    else:
        st.success("✅ Traffic is expected to be normal at your selected time.")

    # --- Visualization: Actual vs Predicted ---
    if df is not None:
        # Filter actual data for selected junction and date
        actual = df[(df['Junction'] == junction) & (df['DateTime'].dt.date == date)]
        if not actual.empty:
            st.subheader(f"Actual vs Predicted for {junction} on {date}")
            actual_hourly = actual.groupby(actual['DateTime'].dt.hour)['Vehicles'].mean()
            hours = list(range(24))
            predicted = []
            for h in hours:
                # Prepare features for each hour
                hour_sin = np.sin(2 * np.pi * h / 24)
                hour_cos = np.cos(2 * np.pi * h / 24)
                weekday = dt.weekday()
                weekday_sin = np.sin(2 * np.pi * weekday / 7)
                weekday_cos = np.cos(2 * np.pi * weekday / 7)
                input_df = pd.DataFrame([{
                    'Junction_enc': junction_enc,
                    'Year': year,
                    'Month': month,
                    'Day': day,
                    'Hour': h,
                    'Weekday': weekday,
                    'Hour_sin': hour_sin,
                    'Hour_cos': hour_cos,
                    'Weekday_sin': weekday_sin,
                    'Weekday_cos': weekday_cos
                }])
                pred = model.predict(input_df)[0]
                predicted.append(pred)
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=actual_hourly.values, mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=hours, y=predicted, mode='lines+markers', name='Predicted'))
            fig.update_layout(title=f"Actual vs Predicted Traffic ({junction}, {date})",
                              xaxis_title="Hour of Day", yaxis_title="Vehicle Count")
            st.plotly_chart(fig)
        else:
            st.info("No recorded traffic data for this junction and date. Try another date or junction.")
    else:
        st.info("Actual data file not found. Place traffic.csv in the data folder for visualizations.")

# --- General Traffic Trends ---
if df is not None:
    st.subheader("Overall Traffic Trends")

    # Select junction for filtering
    selected_junction = st.selectbox("Show trends for junction:", df['Junction'].unique())
    filtered = df[df['Junction'] == selected_junction]

    # Average vehicle count by hour for selected junction
    filtered['hour'] = filtered['DateTime'].dt.hour
    hourly_avg = filtered.groupby('hour')['Vehicles'].mean()

    # Find top 3 peak hours
    top_hours = hourly_avg.sort_values(ascending=False).head(3)

    # Plot with annotations using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_avg.index, y=hourly_avg.values, mode='lines+markers', name='Avg Vehicles'))

    # Annotate peak hours
    for h, cnt in top_hours.items():
        fig.add_annotation(
            x=h, y=cnt,
            text=f"🔥 Peak: {int(cnt)}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            bgcolor="orange"
        )

    # Annotate special event/holiday if selected date is a holiday
    selected_hour = time.hour
    if is_holiday and selected_hour in hourly_avg.index:
        fig.add_annotation(
            x=selected_hour, y=hourly_avg[selected_hour],
            text="🛕 Festival/Holiday",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-60,
            bgcolor="yellow"
        )
    fig.update_layout(
        title=f"Traffic Trends for {selected_junction}",
        xaxis_title="Hour of Day",
        yaxis_title="Average Vehicle Count"
    )
    st.plotly_chart(fig)
else:
    st.info("Place traffic.csv in the data folder to see overall traffic trends.")