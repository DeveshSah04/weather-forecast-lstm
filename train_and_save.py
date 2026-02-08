import requests
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("models", exist_ok=True)

# Major Mumbai Local Stations (Western + Central + Harbour)
stations = [
    {"name":"Churchgate","lat":18.935,"lon":72.826},
    {"name":"Marine_Lines","lat":18.943,"lon":72.823},
    {"name":"Mumbai_Central","lat":18.969,"lon":72.820},
    {"name":"Dadar","lat":19.0178,"lon":72.8478},
    {"name":"Bandra","lat":19.0596,"lon":72.8295},
    {"name":"Santacruz","lat":19.081,"lon":72.842},
    {"name":"Vile_Parle","lat":19.100,"lon":72.843},
    {"name":"Andheri","lat":19.1136,"lon":72.8697},
    {"name":"Goregaon","lat":19.155,"lon":72.849},
    {"name":"Malad","lat":19.187,"lon":72.848},
    {"name":"Kandivali","lat":19.205,"lon":72.850},
    {"name":"Borivali","lat":19.2307,"lon":72.8567}
]

SEQ_LENGTH = 7
EPOCHS = 30
HISTORICAL_DAYS = 3650

def fetch_data(lat, lon):
    end = datetime.now().date()-timedelta(days=1)
    start = end - timedelta(days=HISTORICAL_DAYS)

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_mean&timezone=auto"

    data = requests.get(url).json()

    df = pd.DataFrame({
        "temp_mean": data["daily"]["temperature_2m_mean"]
    }).ffill().bfill()

    return df

def create_sequences(data):
    X,y=[],[]
    for i in range(len(data)-SEQ_LENGTH):
        X.append(data[i:i+SEQ_LENGTH])
        y.append(data[i+SEQ_LENGTH])
    return np.array(X),np.array(y)

def build_model():
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH,1))),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

for stn in stations:

    print("Training:", stn["name"])

    df = fetch_data(stn["lat"], stn["lon"])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X,y = create_sequences(scaled)

    model = build_model()

    model.fit(X,y,
              epochs=EPOCHS,
              batch_size=32,
              callbacks=[EarlyStopping(patience=5)],
              verbose=1)

    model.save(f"models/{stn['name']}_model.h5")
    joblib.dump(scaler, f"models/{stn['name']}_scaler.pkl")

print("ALL STATION MODELS TRAINED")
