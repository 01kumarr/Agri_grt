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
df = load_data('data_for_lstm_model3.csv')
print(df.index.max())
scaler = load_scaler('robust_scaler3.pkl')
model = load_keras_model('best_model_with_kerastuner3.keras')
lookbacks = 4

def prepare_input(data, lookbacks):
    input_data = []
    for i in range(len(data)-lookbacks+1):
        input_data.append(data[i:i + lookbacks, :])
    return np.array(input_data)

def predict_future(data, model, scaler, window_size=4, future_steps=10, lookbacks=4):

    scaled_data = scaler.transform(data[-window_size:]) 
    sliced_data = scaled_data[-lookbacks:] 
    input_data = prepare_input(sliced_data, lookbacks)

    predictions = []

    for _ in range(future_steps):
        y_pred = model.predict(input_data)
        y_pred *= 1.1137  # Increase predictions by 5%
        predictions.append(y_pred[0])  # Appending the adjusted prediction array

        sliced_data = np.append(sliced_data, y_pred, axis=0)
        sliced_data = sliced_data[-lookbacks:]
        input_data = prepare_input(sliced_data, lookbacks)

    return np.array(predictions)

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
            predictions = predict_future(df.values, model, scaler, window_size=100, 
                                         future_steps=future_steps, lookbacks=lookbacks)
            
            # Extract only the first feature from predictions
            predictions_first_feature = predictions[:, 0].reshape(-1, 1)

            # Perform inverse scaling only on the first feature
            future_prices = scaler.inverse_transform(np.hstack([predictions_first_feature, 
                                                                np.zeros((predictions_first_feature.shape[0], scaler.scale_.shape[0]-1))]))[:, 0]

            print(future_prices)
            # Create future dates
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_steps + 1)]
            
            # Plot historical data and future predictions
            last_100 = df.iloc[-50:]
            fig, ax = plt.subplots()
            ax.plot(last_100.index, last_100.iloc[:, 0], label='Historical Data')  # Only plot the first feature
            ax.plot(future_dates, future_prices * 1.05, 'r--', label='Future Predictions')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Historical Data and Future Predictions')
            ax.legend()
            

            # Rotate date labels and adjust layout
            fig.autofmt_xdate(rotation=45)
            plt.grid()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.write(f"Predicted price for {selected_date} is: {future_prices[-1]:.2f}")
        else:
            st.write("Selected date is not in the future. Please select a future date.")
    except Exception as e:
        st.error(f"Error: {e}")
