# Stock Price Prediction using LSTM

This repository contains a Python script to predict stock prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on historical stock price data and uses technical indicators as features.

## Features
- Data Collection: Downloads historical stock data using `yfinance`.
- Data Preprocessing: Computes technical indicators such as SMA, RSI, Momentum, and Volatility.
- LSTM Model: Constructs and trains an LSTM model for predicting stock prices.
- Model Evaluation: Evaluates the model using RMSE and plots actual vs. predicted stock prices.

## Installation

To set up your environment and install the required packages, create a virtual environment and use the `requirements.txt` file.

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

1. **Run the script:**

    ```bash
    python stock_prediction.py
    ```

    Ensure that the script file is named `stock_prediction.py` or update the command to reflect the actual script name.

2. **Visualize Results:**

    The script will output the Root Mean Squared Error (RMSE) and display a plot comparing actual and predicted stock prices.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance)
- [TA-Lib](https://github.com/bukosabino/ta)
- [TensorFlow](https://www.tensorflow.org/)
