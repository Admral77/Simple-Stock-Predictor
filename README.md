# Stock Price Prediction using LSTM

This repository contains a Python script that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The script is designed to be flexible, allowing users to easily modify parameters and test different stocks.

## Features
- **Data Collection:** Downloads historical stock data using `yfinance`.
- **Data Preprocessing:** Computes technical indicators such as Simple Moving Averages (SMA), Relative Strength Index (RSI), Momentum, and Volatility.
- **LSTM Model:** Constructs and trains an LSTM model for predicting stock prices.
- **Model Evaluation:** Evaluates the model using Root Mean Squared Error (RMSE) and plots the actual vs. predicted stock prices.

## Installation

To set up your environment and install the required packages, follow these steps:

1. **Create a virtual environment:**

    ```bash
    python -m venv env
    ```

2. **Activate the virtual environment:**

    - On Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source env/bin/activate
      ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Edit the Script:**

    Open `StockPredict.py` in your preferred text editor or IDE. You can modify the following parameters:
    
    - **`ticker`:** Change this variable to the stock symbol you want to analyze. For example, use `"GOOGL"` for Alphabet Inc.
    - **Date Range:** Adjust the `start` and `end` dates in the `yf.download()` function to fit the time period you want to analyze.
    - **Technical Indicators:** You can modify the parameters for the technical indicators, such as the window sizes for SMA, RSI, and others.

2. **Run the Script:**

    Execute the script using the following command:

    ```bash
    python StockPredict.py
    ```

    This will:
    - Download the stock data for the specified ticker and date range.
    - Preprocess the data and compute the technical indicators.
    - Train an LSTM model on the data.
    - Evaluate the model's performance and display a plot comparing actual and predicted stock prices.

3. **Customizing the Model:**

    - **LSTM Configuration:** You can adjust the number of LSTM units, dropout rates, and other parameters in the model architecture.
    - **Training Parameters:** Modify the number of epochs, batch size, and early stopping criteria to optimize the training process.

4. **Visualizing Results:**

    After running the script, a plot will be displayed showing the actual vs. predicted stock prices. You can save or customize this plot by modifying the plotting section of the script.

## Example

Here’s a quick example of how to change the stock ticker and date range:

```python
ticker = "GOOGL"  # Change to your desired stock symbol
stock_data = yf.download(ticker, start="2015-01-01", end="2022-01-01")
