import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Weather LSTM Forecasting",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'locations_data' not in st.session_state:
    st.session_state.locations_data = {}
if 'selected_locations' not in st.session_state:
    st.session_state.selected_locations = []

# Functions
@st.cache_data(ttl=3600)
def search_locations(query):
    """Search for locations using geocoding API"""
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=10&language=en&format=json"
        response = requests.get(url)
        data = response.json()
        
        if 'results' in data:
            locations = []
            for result in data['results']:
                location_info = {
                    'name': result['name'],
                    'country': result.get('country', 'Unknown'),
                    'admin1': result.get('admin1', ''),
                    'latitude': result['latitude'],
                    'longitude': result['longitude'],
                    'display_name': f"{result['name']}, {result.get('admin1', '')}, {result.get('country', '')}"
                }
                locations.append(location_info)
            return locations
        return []
    except Exception as e:
        st.error(f"Error searching locations: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_weather_data(latitude, longitude, days=3650):
    try:
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)

        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={latitude}&longitude={longitude}"
            f"&start_date={start_date}&end_date={end_date}"
            "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,windspeed_10m_max"
            "&timezone=auto"
        )

        response = requests.get(url, timeout=30)
        data = response.json()

        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temp_max': data['daily']['temperature_2m_max'],
            'temp_min': data['daily']['temperature_2m_min'],
            'temp_mean': data['daily']['temperature_2m_mean'],
            'precipitation': data['daily']['precipitation_sum'],
            'windspeed': data['daily']['windspeed_10m_max'],
            'humidity': [50] * len(data['daily']['time'])
        })

        df = df.sort_values('date').ffill().bfill()
        return df

    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None


@st.cache_data(ttl=1800)
def fetch_hourly_weather_data(latitude, longitude, forecast_days=7):
    """Fetch hourly weather forecast data from Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover&forecast_days={forecast_days}&timezone=auto"
        
        response = requests.get(url)
        data = response.json()
        
        # Check if we got an error response
        if 'hourly' not in data:
            st.error(f"API Error: {data.get('reason', 'Unknown error')}")
            return None
        
        # FIXED: Use 'hourly' for forecast API
        df = pd.DataFrame({
            'datetime': pd.to_datetime(data['hourly']['time']),
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'precipitation': data['hourly']['precipitation'],
            'windspeed': data['hourly']['wind_speed_10m'],
            'cloud_cover': data['hourly']['cloud_cover']
        })
        
        # Handle NaN values - CRITICAL FIX
        nan_counts = df.isna().sum()
        if nan_counts.any() and nan_counts.sum() > 0:
            nan_info = {col: count for col, count in nan_counts.items() if count > 0}
            st.warning(f"Found missing values in {len(nan_info)} columns: {nan_info}")
        
        # Fill NaN values using forward fill, then backward fill
        df = df.ffill().bfill()
        
        # If still NaN (entire column is NaN), fill with reasonable defaults
        if df['temperature'].isna().any():
            df['temperature'] = df['temperature'].fillna(20)
        if df['humidity'].isna().any():
            df['humidity'] = df['humidity'].fillna(50)
        if df['precipitation'].isna().any():
            df['precipitation'] = df['precipitation'].fillna(0)
        if df['windspeed'].isna().any():
            df['windspeed'] = df['windspeed'].fillna(10)
        if df['cloud_cover'].isna().any():
            df['cloud_cover'] = df['cloud_cover'].fillna(50)
        
        # Final check - drop any remaining rows with NaN in critical columns
        critical_cols = ['datetime', 'temperature']
        before_drop = len(df)
        df = df.dropna(subset=critical_cols)
        if len(df) < before_drop:
            st.info(f"Dropped {before_drop - len(df)} rows with missing critical data")
        
        # Ensure we have enough data
        if len(df) < 10:
            st.error(f"Insufficient data after cleaning: only {len(df)} rows available")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error fetching hourly weather data: {e}")
        return None

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, n_features=1):
    """Build LSTM model architecture"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(seq_length, n_features))),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_features)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(df, seq_length, epochs):
    try:
        data = df[['temp_mean']].values

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        X, y = create_sequences(scaled, seq_length)

        tscv = TimeSeriesSplit(n_splits=5)

        mae_scores, rmse_scores, adj_r2_scores = [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = build_lstm_model(seq_length, 1)

            es = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                callbacks=[es],
                verbose=0
            )

            y_pred = model.predict(X_test, verbose=0)

            y_test_inv = scaler.inverse_transform(y_test)
            y_pred_inv = scaler.inverse_transform(y_pred)

            mae_scores.append(mean_absolute_error(y_test_inv, y_pred_inv))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))

            r2 = r2_score(y_test_inv, y_pred_inv)
            adj_r2_scores.append(
                adjusted_r2_score(r2, len(y_test_inv), 1)
            )

        metrics = {
            'mae': np.mean(mae_scores),
            'rmse': np.mean(rmse_scores),
            'adj_r2': np.nanmean(adj_r2_scores),
            'history': history.history
        }

        return model, scaler, metrics, X_test, y_test, y_pred

    except Exception as e:
        st.error(f"LSTM training error: {e}")
        return None, None, None, None, None, None

def generate_forecast(model, scaler, last_data, seq_length, forecast_days):
    """Generate future predictions"""
    try:
        predictions = []
        current_sequence = last_data.copy()
        
        # Validate input
        if np.isnan(current_sequence).any() or np.isinf(current_sequence).any():
            st.error("Invalid data for forecasting")
            return None
        
        for _ in range(forecast_days):
            pred = model.predict(current_sequence.reshape(1, seq_length, 1), verbose=0)
            
            # Check for invalid predictions
            if np.isnan(pred).any() or np.isinf(pred).any():
                st.warning("Model produced invalid prediction, stopping forecast")
                break
            
            predictions.append(pred[0, 0])
            current_sequence = np.append(current_sequence[1:], pred)
        
        if len(predictions) == 0:
            return None
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Final validation
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            st.error("Forecast produced invalid values")
            return None
        
        return predictions.flatten()
    
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        return None
    
def adjusted_r2_score(r2, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# Main App
st.title("Weather Forecasting")

# Create main tabs
main_tab1, main_tab2 = st.tabs(["Daily Forecast (LSTM)", "Hourly Forecast"])

with main_tab1:

# Sidebar
    st.sidebar.header("Configuration")

# Location Search
    st.sidebar.subheader("Search Locations")
    search_query = st.sidebar.text_input("Search for a city", placeholder="e.g., Mumbai")

    if search_query:
        with st.spinner("Searching locations..."):
            locations = search_locations(search_query)
        
        if locations:
            st.sidebar.success(f"Found {len(locations)} locations")
            
            # Display location options
            for loc in locations:
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.sidebar.write(f"{loc['display_name']}")
                with col2:
                    if st.sidebar.button("Add", key=f"add_{loc['name']}_{loc['latitude']}"):
                        if loc['display_name'] not in st.session_state.selected_locations:
                            st.session_state.selected_locations.append(loc['display_name'])
                            st.session_state.locations_data[loc['display_name']] = loc
                            st.sidebar.success("Added!")
                        else:
                            st.sidebar.warning("Already added")
        else:
            st.sidebar.info("No locations found. Try a different search term.")

    # Display selected locations
    st.sidebar.subheader("Selected Locations")
    if st.session_state.selected_locations:
        for location in st.session_state.selected_locations:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.sidebar.write(f"✓ {location}")
            with col2:
                if st.sidebar.button("❌", key=f"remove_{location}"):
                    st.session_state.selected_locations.remove(location)
                    del st.session_state.locations_data[location]
                    st.rerun()
    else:
        st.sidebar.info("No locations selected yet")

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    seq_length = st.sidebar.slider("Sequence Length (days)", 3, 21, 7, 
                                help="Number of past days to use for prediction")
    forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7,
                                    help="Number of days to predict")
    epochs = st.sidebar.slider("Training Epochs", 20, 100, 50,
                            help="Number of training iterations")
    historical_days = st.sidebar.slider("Historical Data (days)",365, 3650, 3650,
                            help="Amount of historical data to fetch (up to 10 years)"
)


    st.sidebar.subheader("Hourly Forecast")
    hourly_forecast_days = st.sidebar.slider("Hourly Forecast Days", 1, 7, 3,
                                            help="Number of days for hourly forecast")

    # Filter options
    st.sidebar.subheader("Filter Options")
    location_filter = st.sidebar.multiselect(
        "Select locations to display",
        options=st.session_state.selected_locations,
        default=st.session_state.selected_locations
    )

    # Main content - Daily Forecast Tab
    if st.session_state.selected_locations:
        
        # Fetch and Train button
        if st.sidebar.button("Fetch Data & Train Model", type="primary"):
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_locations = len(st.session_state.selected_locations)
            
            for idx, location_name in enumerate(st.session_state.selected_locations):
                location = st.session_state.locations_data[location_name]
                
                status_text.text(f"Processing {location_name} ({idx + 1}/{total_locations})...")
                
                # Fetch data
                df = fetch_weather_data(location['latitude'], location['longitude'], historical_days)
                
                if df is not None and len(df) > seq_length:
                    # Train model

                    model_path = f"models/{location['name']}_model.h5"
                    scaler_path = f"models/{location['name']}_scaler.pkl"

                    if os.path.exists(model_path) and os.path.exists(scaler_path):

                        model = load_model(model_path, compile=False)
                        scaler = joblib.load(scaler_path)

                        # Fetch latest data
                        df = fetch_weather_data(location['latitude'], location['longitude'], historical_days)

                        scaled_data = scaler.transform(df[['temp_mean']].values)
                        last_sequence = scaled_data[-seq_length:]

                        predictions = generate_forecast(model, scaler, last_sequence, seq_length, forecast_days)

                        # Calculate metrics using last 30 days

                        test_actual = df[['temp_mean']].values[-30:]
                        scaled_test = scaler.transform(test_actual)

                        X_test_seq, y_test_seq = create_sequences(scaled_test, seq_length)

                        y_pred_seq = model.predict(X_test_seq)

                        y_test_inv = scaler.inverse_transform(y_test_seq)
                        y_pred_inv = scaler.inverse_transform(y_pred_seq)

                        mae = mean_absolute_error(y_test_inv, y_pred_inv)
                        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
                        r2 = r2_score(y_test_inv, y_pred_inv)
                        adj_r2 = adjusted_r2_score(r2, len(y_test_inv), 1)

                        metrics = {
                            "mae": mae,
                            "rmse": rmse,
                            "adj_r2": adj_r2,
                            "history":{"loss":[0],"val_loss":[0],"mae":[0],"val_mae":[0]}
                        }


                        if predictions is not None:
                            results[location_name] = {
                                'df': df,
                                'model': model,
                                'scaler': scaler,
                                'metrics': metrics,
                                'predictions': predictions,
                                'location': location
                            }

                    else:
                        st.error(f"Model not found for {location['name']}. Train offline first.")
                
                progress_bar.progress((idx + 1) / total_locations)
            
            st.session_state.results = results
            status_text.text("✅ All locations processed!")
            
            if len(results) > 0:
                st.success(f"Successfully trained models for {len(results)} locations!")
            else:
                st.error("No locations were successfully processed. Check the warnings above.")
        
        # Display results
        if 'results' in st.session_state and st.session_state.results:
            
            # Filter results based on selection
            filtered_results = {k: v for k, v in st.session_state.results.items() 
                            if k in location_filter}
            
            if filtered_results:
                # Metrics Overview
                
                #st.header("Model Performance Metrics")
                
                #cols = st.columns(len(filtered_results))
                #for idx, (location_name, data) in enumerate(filtered_results.items()):
                    #with cols[idx]:
                        #st.subheader(location_name.split(',')[0])
                        #st.metric("MAE", f"{data['metrics']['mae']:.2f}°C")
                        #st.metric("RMSE", f"{data['metrics']['rmse']:.2f}°C")
                        #st.metric("Adjusted R²", f"{data['metrics']['adj_r2']:.3f}")
                       
                # Comparison Chart
                st.header("Temperature Comparison")
                
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=["Multi-Location Temperature Forecast"]
                )
                
                colors = px.colors.qualitative.Set3
                
                for idx, (location_name, data) in enumerate(filtered_results.items()):
                    df = data['df']
                    predictions = data['predictions']
                    color = colors[idx % len(colors)]
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['temp_mean'],
                        mode='lines',
                        name=f"{location_name.split(',')[0]} - Historical",
                        line=dict(color=color, width=2)
                    ))
                    
                    # Forecast
                    future_dates = pd.date_range(
                        start=df['date'].iloc[-1] + timedelta(days=1),
                        periods=forecast_days
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        mode='lines+markers',
                        name=f"{location_name.split(',')[0]} - Forecast",
                        line=dict(color=color, width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Temperature (°C)",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed forecasts
                st.header("Detailed Forecasts")
                
                tabs = st.tabs([loc.split(',')[0] for loc in filtered_results.keys()])
                
                for idx, (location_name, data) in enumerate(filtered_results.items()):
                    with tabs[idx]:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Individual forecast chart
                            df = data['df']
                            predictions = data['predictions']
                            
                            fig_individual = go.Figure()
                            
                            # Historical
                            fig_individual.add_trace(go.Scatter(
                                x=df['date'],
                                y=df['temp_mean'],
                                mode='lines',
                                name='Historical',
                                line=dict(color='#3b82f6', width=2)
                            ))
                            
                            # Forecast
                            future_dates = pd.date_range(
                                start=df['date'].iloc[-1] + timedelta(days=1),
                                periods=forecast_days
                            )
                            
                            fig_individual.add_trace(go.Scatter(
                                x=future_dates,
                                y=predictions,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='#10b981', width=2, dash='dash'),
                                marker=dict(size=8)
                            ))
                            
                            fig_individual.update_layout(
                                title=f"Temperature Forecast - {location_name}",
                                xaxis_title="Date",
                                yaxis_title="Temperature (°C)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_individual, use_container_width=True)
                        
                        with col2:
                            st.subheader("Forecast Table")
                            forecast_df = pd.DataFrame({
                                'Date': future_dates.strftime('%Y-%m-%d'),
                                'Temperature (°C)': [f"{temp:.1f}" for temp in predictions]
                            })
                            st.dataframe(forecast_df, use_container_width=True, height=400)
                        
                        # Recent historical data
                        st.subheader("Recent Historical Data")
                        recent_df = df[['date', 'temp_mean', 'temp_max', 'temp_min', 
                                    'precipitation', 'windspeed', 'humidity']].tail(7)
                        recent_df.columns = ['Date', 'Avg Temp (°C)', 'Max Temp (°C)', 
                                            'Min Temp (°C)', 'Precipitation (mm)', 
                                            'Wind Speed (km/h)', 'Humidity (%)']
                        st.dataframe(recent_df, use_container_width=True)
                        
                        # Download forecast
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast CSV",
                            data=csv,
                            file_name=f"{location_name.split(',')[0]}_forecast.csv",
                            mime='text/csv'
                        )
                
                # Training history
                st.header("📉 Training History")
                
                for location_name, data in filtered_results.items():
                    with st.expander(f"View training history - {location_name}"):
                        history = data['metrics']['history']
                        
                        fig_history = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('Loss', 'Mean Absolute Error')
                        )
                        
                        # Loss
                        fig_history.add_trace(go.Scatter(
                            y=history['loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='#3b82f6')
                        ), row=1, col=1)
                        
                        fig_history.add_trace(go.Scatter(
                            y=history['val_loss'],
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color='#ef4444')
                        ), row=1, col=1)
                        
                        # MAE
                        fig_history.add_trace(go.Scatter(
                            y=history['mae'],
                            mode='lines',
                            name='Training MAE',
                            line=dict(color='#10b981')
                        ), row=1, col=2)
                        
                        fig_history.add_trace(go.Scatter(
                            y=history['val_mae'],
                            mode='lines',
                            name='Validation MAE',
                            line=dict(color='#f59e0b')
                        ), row=1, col=2)
                        
                        fig_history.update_xaxes(title_text="Epoch")
                        fig_history.update_layout(height=400)
                        
                        st.plotly_chart(fig_history, use_container_width=True)
            
            else:
                st.warning("No locations selected in filter. Please select at least one location.")

# Hourly Forecast Tab
with main_tab2:
    st.header("Hourly Weather Forecast")
    st.markdown("Real-time hourly weather predictions for the next few days")
    
    if st.session_state.selected_locations:
        
        if st.button("Fetch Hourly Forecast", type="primary"):
            hourly_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_locations = len(st.session_state.selected_locations)
            
            for idx, location_name in enumerate(st.session_state.selected_locations):
                location = st.session_state.locations_data[location_name]
                
                status_text.text(f"Fetching hourly data for {location_name} ({idx + 1}/{total_locations})...")
                
                # Fetch hourly data
                hourly_df = fetch_hourly_weather_data(location['latitude'], location['longitude'], hourly_forecast_days)
                
                if hourly_df is not None:
                    hourly_results[location_name] = {
                        'df': hourly_df,
                        'location': location
                    }
                
                progress_bar.progress((idx + 1) / total_locations)
            
            st.session_state.hourly_results = hourly_results
            status_text.text("✅ Hourly data fetched!")
            st.success(f"Successfully fetched hourly forecasts for {len(hourly_results)} locations!")
        
        # Display hourly results
        if 'hourly_results' in st.session_state and st.session_state.hourly_results:
            
            # Filter results
            filtered_hourly_results = {k: v for k, v in st.session_state.hourly_results.items() 
                                      if k in location_filter}
            
            if filtered_hourly_results:
                
                # Time interval selector
                time_intervals = {
                    "Every Hour": 1,
                    "Every 3 Hours": 3,
                    "Every 6 Hours": 6,
                    "Every 12 Hours": 12
                }
                
                selected_interval = st.selectbox("Select Time Interval", list(time_intervals.keys()))
                interval_hours = time_intervals[selected_interval]
                
                # Multi-location comparison
                st.subheader("Temperature Comparison (Hourly)")
                
                fig_temp = go.Figure()
                colors = px.colors.qualitative.Set2
                
                for idx, (location_name, data) in enumerate(filtered_hourly_results.items()):
                    df = data['df']
                    # Filter by interval
                    df_filtered = df.iloc[::interval_hours]
                    
                    fig_temp.add_trace(go.Scatter(
                        x=df_filtered['datetime'],
                        y=df_filtered['temperature'],
                        mode='lines+markers',
                        name=location_name.split(',')[0],
                        line=dict(color=colors[idx % len(colors)], width=2),
                        marker=dict(size=6)
                    ))
                
                fig_temp.update_layout(
                    height=500,
                    xaxis_title="Date & Time",
                    yaxis_title="Temperature (°C)",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Detailed hourly forecasts per location
                st.subheader("Detailed Hourly Forecasts by Location")
                
                tabs = st.tabs([loc.split(',')[0] for loc in filtered_hourly_results.keys()])
                
                for idx, (location_name, data) in enumerate(filtered_hourly_results.items()):
                    with tabs[idx]:
                        df = data['df']
                        df_filtered = df.iloc[::interval_hours].copy()
                        
                        # Add day of week and time
                        df_filtered['day'] = df_filtered['datetime'].dt.strftime('%A, %b %d')
                        df_filtered['time'] = df_filtered['datetime'].dt.strftime('%I:%M %p')
                        
                        # Create multi-metric chart
                        fig_multi = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Temperature', 'Humidity', 'Precipitation', 'Wind Speed'),
                            vertical_spacing=0.12,
                            horizontal_spacing=0.1
                        )
                        
                        # Temperature
                        fig_multi.add_trace(go.Scatter(
                            x=df_filtered['datetime'],
                            y=df_filtered['temperature'],
                            mode='lines+markers',
                            name='Temperature',
                            line=dict(color='#ef4444', width=2),
                            marker=dict(size=6),
                            showlegend=False
                        ), row=1, col=1)
                        
                        # Humidity
                        fig_multi.add_trace(go.Scatter(
                            x=df_filtered['datetime'],
                            y=df_filtered['humidity'],
                            mode='lines+markers',
                            name='Humidity',
                            line=dict(color='#3b82f6', width=2),
                            marker=dict(size=6),
                            showlegend=False
                        ), row=1, col=2)
                        
                        # Precipitation
                        fig_multi.add_trace(go.Bar(
                            x=df_filtered['datetime'],
                            y=df_filtered['precipitation'],
                            name='Precipitation',
                            marker=dict(color='#06b6d4'),
                            showlegend=False
                        ), row=2, col=1)
                        
                        # Wind Speed
                        fig_multi.add_trace(go.Scatter(
                            x=df_filtered['datetime'],
                            y=df_filtered['windspeed'],
                            mode='lines+markers',
                            name='Wind Speed',
                            line=dict(color='#10b981', width=2),
                            marker=dict(size=6),
                            showlegend=False
                        ), row=2, col=2)
                        
                        fig_multi.update_xaxes(title_text="Time", row=1, col=1)
                        fig_multi.update_xaxes(title_text="Time", row=1, col=2)
                        fig_multi.update_xaxes(title_text="Time", row=2, col=1)
                        fig_multi.update_xaxes(title_text="Time", row=2, col=2)
                        
                        fig_multi.update_yaxes(title_text="°C", row=1, col=1)
                        fig_multi.update_yaxes(title_text="%", row=1, col=2)
                        fig_multi.update_yaxes(title_text="mm", row=2, col=1)
                        fig_multi.update_yaxes(title_text="km/h", row=2, col=2)
                        
                        fig_multi.update_layout(height=600)
                        
                        st.plotly_chart(fig_multi, use_container_width=True)
                        
                        # Hourly data table with time slots
                        st.subheader("Hourly Forecast Table")
                        
                        # Group by day for better readability
                        for day in df_filtered['day'].unique()[:3]:  # Show first 3 days
                            day_data = df_filtered[df_filtered['day'] == day]
                            
                            with st.expander(f"{day}", expanded=True):
                                display_df = pd.DataFrame({
                                    'Time': day_data['time'],
                                    'Temp (°C)': day_data['temperature'].apply(lambda x: f"{x:.1f}"),
                                    'Humidity (%)': day_data['humidity'].apply(lambda x: f"{x:.0f}"),
                                    'Precip (mm)': day_data['precipitation'].apply(lambda x: f"{x:.1f}"),
                                    'Wind (km/h)': day_data['windspeed'].apply(lambda x: f"{x:.1f}"),
                                    'Cloud (%)': day_data['cloud_cover'].apply(lambda x: f"{x:.0f}")
                                })
                                st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv = df_filtered[['datetime', 'temperature', 'humidity', 'precipitation', 
                                          'windspeed', 'cloud_cover']].to_csv(index=False)
                        st.download_button(
                            label="📥 Download Hourly Forecast CSV",
                            data=csv,
                            file_name=f"{location_name.split(',')[0]}_hourly_forecast.csv",
                            mime='text/csv'
                        )
                
            else:
                st.warning("No locations selected in filter. Please select at least one location.")
        
#python -m venv venv
#.\venv\Scripts\Activate.ps1
#Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#streamlit run weather.py
#pip install streamlit spotipy tensorflow scikit-learn plotly pandas numpy
#pip uninstall ml-dtypes -y
#pip install ml-dtypes==0.5.0
#pip install tensorflow==2.20.0
#pip install tensorflow==2.20.0 protobuf==4.25.3
