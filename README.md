# Price Prediction Using LSTM with Attention
This repository contains three models designed to predict crop prices using LSTM (Long Short-Term Memory) neural networks with an Attention mechanism. These models are created and tuned using the Keras library and Keras Tuner for hyperparameter optimization.

### Project Overview
The main objective of this project is to predict the future prices of crops based on historical data. The models use LSTM layers combined with Attention mechanisms to focus on relevant parts of the data. The project includes custom components like lambda functions, callbacks, and hyperparameter tuning.

#### Features:
LSTM layers: Handle the sequential nature of time-series data.
Attention mechanism: Helps the model focus on important parts of the data.
Hyperparameter Tuning: Implemented using Keras Tuner to find the best model parameters.
Custom Callbacks: Print the hyperparameters and predictions during training.
Custom Lambda Functions: Custom TensorFlow operations for data manipulation.

#### Requirements
pip install tensorflow keras keras-tuner numpy pandas scikit-learn

#### Model 1: Basic LSTM with Attention
This model is a simple implementation using LSTM and Attention layers. It includes a custom lambda function to normalize attention outputs and is designed as the base version to evaluate the effectiveness of Attention in LSTM-based models.

Custom Lambda Function for normalizing the attention outputs.
Stacked LSTM layers for sequential data handling.
Attention Layer to focus on relevant parts of the input sequence.
Saved in the model_1.py file.


#### Model 2: Enhanced LSTM-Attention with Hyperparameter Tuning
Building on Model 1, this version incorporates hyperparameter tuning using Keras Tuner. It optimizes key parameters such as the number of LSTM units, dropout rate, and learning rate. Additionally, it includes a custom callback that prints hyperparameters and predictions at the end of each epoch.

Hyperparameter Tuning: Optimize LSTM units, dropout rate, learning rate, and batch size.
Custom Callback: Logs predictions and hyperparameters during training.
Saved in the model_2.py file.

#### Model 3: Advanced LSTM-Attention with RMSprop Optimizer
This advanced version of the LSTM-Attention model uses the RMSprop optimizer for better convergence in time-series tasks. The model adds more flexibility in hyperparameter tuning, including tuning the RMSprop-specific parameters like rho, momentum, and epsilon. A more detailed callback logs both hyperparameters and predictions.

RMSprop Optimizer for different optimization strategies.
Expanded Hyperparameter Tuning including RMSprop-specific parameters.
Custom Callbacks to log hyperparameters and predictions.
Saved in the model_3.py file.

#### Future Work
Explore other Attention mechanisms to further improve performance.
Add more features to capture seasonality and market conditions in price predictions.
Experiment with additional optimizers like Nadam or custom optimization strategies.



## Project Structure

```bash
├── data_visualization.py      # Code for visualizing data
├── data_handling.py           # Code for handling and preprocessing data
├── model1/                    # Folder for the first model
│   ├── base_model.py          # Script for training and saving model1
│   ├── app.py                 # Streamlit app for running model1
│   ├── model1_weights.pk      # MinMaxScaler and model weights for model1
│   ├── real_vs_predicted.png  # Real vs predicted price plot for model1
│   ├── residual_plot.png      # Residual plot for model1
├── model2/                    # Folder for the second model (similar structure)
│   ├── base_model.py
│   ├── app.py
│   ├── model2_weights.pk
│   ├── real_vs_predicted.png
│   ├── residual_plot.png
├── model3/                    # Folder for the third model (similar structure)
│   ├── base_model.py
│   ├── app.py
│   ├── model3_weights.pk
│   ├── real_vs_predicted.png
│   ├── residual_plot.png
└── README.md 
