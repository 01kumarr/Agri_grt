import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# Cache the loading of historical data
@st.cache_data
def load_data(file_name):
    df = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    df.sort_index(inplace=True)
    return df

# Cache the loading of the scaler
@st.cache_data
def load_scaler(file_name):
    with open(file_name, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Cache the loading of the model
@st.cache_resource
def load_keras_model(model_path):
    return load_model(model_path)

# Load data, scaler, and model
df = load_data('data_for_lstm_model2.csv')
scaler = load_scaler('minmax_scaler2.pkl')
model = load_keras_model('best_model_with_kerastuner2.keras')
lookbacks = 7
rolling_size = 7

print(df)

def prepare_future_features(data = df, window_size=10):
    if len(data) < window_size:
        window_size = len(data)
    if rolling_size > window_size:
        window_size = rolling_size + 1

    last_window_data = data.iloc[-window_size:]  # slice last 100 data
    rolling_data = last_window_data.rolling(window=7).mean()

    return rolling_data

def prepare_input(data, scaler, lookbacks=4):
    # Transform the data using the scaler
    scaled_data = scaler.transform(data)  # Scaled data is of shape (100, 21)
    
    # Remove the target feature after scaling
    scaled_data = scaled_data[:, :-1]  # Now the shape is (100, 20)
    
    # Prepare the data for LSTM input
    input_data = []
    for i in range(len(scaled_data) - lookbacks + 1):
        input_data.append(scaled_data[i:i + lookbacks])
    
    input_data = np.array(input_data)
    
    return input_data

def create_dummy_data(predictions, num_features=21):
    # Create dummy data with zeros and the number of records equal to the length of predictions
    dummy_data = np.zeros((len(predictions), num_features))
    # Place the predictions in the appropriate column (assuming the target column is the last one)
    dummy_data[:, -1] = predictions
    return dummy_data

def predict_future(data, model, scaler, window_size=4, future_steps=10,
                   rolling_size=7, lookbacks=4):
    historical_data = prepare_future_features(data, window_size=window_size)
        
    # future_data = historical_data.iloc[-rolling_size:]
    d_size = future_steps+lookbacks-1
    future_data = historical_data.iloc[-d_size:]
    # Create rolling mean
    # for i in range(future_steps + lookbacks - 1):
    #     rolling_mean = future_data.iloc[-rolling_size:].mean(axis=0)
    #     new_record = pd.DataFrame([rolling_mean], columns=future_data.columns)
    #     future_data = pd.concat([future_data, new_record], ignore_index=True)

    # future_data = future_data.iloc[rolling_size:]
    
    print(future_data)
    print(future_data.shape)
    predict_data = prepare_input(future_data, scaler, lookbacks=lookbacks)

    y_pred = model.predict(predict_data)
    
    return np.array(y_pred.ravel())

# Streamlit app
st.title("Future Price Prediction App")

# Date input
selected_date = st.date_input("Select a date", datetime.today())

if st.button('Start Prediction'):
    try:
        # Determine the number of future steps
        today = df.index[-1].date()
        days_into_future = (selected_date - today).days
        
        if days_into_future > 0:
            # Predict future prices
            future_steps = days_into_future
            predictions = predict_future(df, model, scaler, window_size=100, future_steps=future_steps,
                                         rolling_size=rolling_size, lookbacks=lookbacks)
            
            # Create dummy data for inverse transformation
            dummy_data = create_dummy_data(predictions, num_features=df.shape[1])
            future_prices = scaler.inverse_transform(dummy_data)[:, -1]
            print(future_prices.shape)
            print(future_prices)
            
            # Create future dates
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_steps + 1)]
            
            # Plot historical data and future predictions
            last_100 = df.iloc[-50:]
            fig, ax = plt.subplots()
            ax.plot(last_100.index, last_100['Modal Price (Rs./Quintal)'], label='Historical Data')
            ax.plot(future_dates, future_prices, 'r--', label='Future Predictions')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Historical Data and Future Predictions')
            ax.legend()
            st.pyplot(fig)
            
            st.write(f"Predicted price for {selected_date} is: {future_prices[-1]:.2f}")
        else:
            st.write("Selected date is not in the future. Please select a future date.")
    except Exception as e:
        st.error(f"Error: {e}")
