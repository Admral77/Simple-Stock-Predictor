import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import ta  # Technical Analysis library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Collection
ticker = "AAPL"  # Example: Apple stock
stock_data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

# Step 2: Data Preprocessing
# Adding Technical Indicators
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()  # Simple Moving Average
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['RSI'] = ta.momentum.rsi(stock_data['Close'], window=14)  # Relative Strength Index
stock_data['Momentum'] = ta.momentum.roc(stock_data['Close'], window=5)  # Rate of Change
stock_data['Volatility'] = ta.volatility.bollinger_hband(stock_data['Close'], window=20) - \
                           ta.volatility.bollinger_lband(stock_data['Close'], window=20)  # Bollinger Bands
stock_data.dropna(inplace=True)

# Select features and target variable
features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'Momentum', 'Volatility']
target = 'Close'

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[features + [target]])

# Step 3: Preparing Data for LSTM
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :-1])  # All features except target
        Y.append(data[i + time_step, -1])  # Target variable (Close price)
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(scaled_data, time_step)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Step 4: LSTM Model Construction
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Step 5: Model Training
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=32, callbacks=[early_stop])

# Step 6: Model Evaluation
predictions = model.predict(X_test)

# Reshape predictions to fit the scaler's expected input
predictions = predictions.reshape(-1, 1)

# Prepare the data for inverse scaling
scaled_predictions = np.zeros((predictions.shape[0], scaled_data.shape[1]))
scaled_predictions[:, -1] = predictions[:, 0]  # Place predictions in the last column

# Perform inverse scaling
original_predictions = scaler.inverse_transform(scaled_predictions)[:, -1]

# Calculating RMSE
rmse = np.sqrt(mean_squared_error(Y_test, original_predictions))
print(f'Root Mean Squared Error: {rmse}')

# Plotting the results
plt.figure(figsize=(14,5))
plt.plot(stock_data.index[-len(original_predictions):], stock_data[target][-len(original_predictions):], color='blue', label='Actual Stock Price')
plt.plot(stock_data.index[-len(original_predictions):], original_predictions, color='red', label='Predicted Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
