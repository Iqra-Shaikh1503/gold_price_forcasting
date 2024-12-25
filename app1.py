import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the trained stacked model
with open('stacked_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a StandardScaler for scaling input data
scaler = StandardScaler()

# Load the test data (this should be the same data used for training the scaler)
test_data = pd.read_csv(r"C:\forcasting gold price\Gold_data (1).csv")  # Update path if necessary

# Create lag features for the test data
for lag in range(1, 6):
    test_data[f'price_lag_{lag}'] = test_data['price'].shift(lag)
test_data.dropna(inplace=True)

# Prepare the test features
X_test = test_data[[f'price_lag_{i}' for i in range(1, 6)]]
y_test = test_data['price']

# Scale the test features
scaler.fit(X_test)  # Fit the scaler on the test data
X_test_scaled = scaler.transform(X_test)

# Define a function to predict the next 30 days of gold prices
def predict_next_30_days(initial_lags, model, scaler):
    predictions = []
    lags = initial_lags.copy()  # Copy to avoid modifying the input

    for day in range(30):
        # Scale the current lagged values
        scaled_lags = scaler.transform(np.array(lags).reshape(1, -1))
        # Predict the next day's price
        predicted_price = model.predict(scaled_lags)[0]
        predictions.append(predicted_price)

        # Update the lags: drop the oldest and append the new predicted price
        lags.pop(0)
        lags.append(predicted_price)

    return predictions

# Streamlit app layout
st.title('Gold Price Prediction for the Next 30 Days')
st.write("Enter the lagged prices (previous 5 days) to predict the gold price for the next 30 days.")

# Input fields for lagged prices
price_lag_1 = st.number_input('Price Lag 1 (most recent):', value=0.0)
price_lag_2 = st.number_input('Price Lag 2:', value=0.0)
price_lag_3 = st.number_input('Price Lag 3:', value=0.0)
price_lag_4 = st.number_input('Price Lag 4:', value=0.0)
price_lag_5 = st.number_input('Price Lag 5:', value=0.0)

# Button to make the prediction for the next 30 days
if st.button('Predict Next 30 Days'):
    initial_lags = [price_lag_1, price_lag_2, price_lag_3, price_lag_4, price_lag_5]
    predicted_prices_30_days = predict_next_30_days(initial_lags, model, scaler)

    # Display the predicted prices for the next 30 days
    st.write("Predicted Gold Prices for the Next 30 Days:")
    for i, price in enumerate(predicted_prices_30_days, start=1):
        st.write(f"Day {i}: {price:.2f}")

    # Calculate the R² score for the model on the test dataset
    y_pred_test = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_test)
    st.write(f'Model R² Score: {r2:.2f} ({r2 * 100:.2f}%)')

