# QuantiQ (Stock Analysis Platform)

**QuantiQ (Stock Analysis Platform)** is an innovative, responsive, and trading-friendly application designed for comprehensive stock data analysis and forecasting. With detailed historical charts (both line and candlestick), technical indicators (SMA, Bollinger Bands, RSI, MACD), and advanced forecasting models (Polynomial and ARIMA), QuantiQ empowers traders with actionable insights. The platform is built with a modern, interactive, and mobile-responsive UI.

## Live Demo

Access the deployed app here: [https://quantiq.streamlit.app/](https://quantiq.streamlit.app/)

## Features

- **Historical Data Visualization:**  
  Choose between a Line Chart or a Candlestick Chart (with volume and a 20-day moving average) to view historical stock data.
  
- **Technical Analysis:**  
  View key indicators such as SMA, Bollinger Bands, RSI, and MACD to analyze market trends.
  
- **Forecast & Predictions:**  
  Forecast future stock prices using advanced Polynomial Regression or ARIMA models. Detailed forecast results are displayed in a table.
  
- **Current & Realtime Data:**  
  See the current market status (open, high, low, price, volume) and realtime updates via Alpha Vantage.
  
- **Responsive Design:**  
  Fully responsive and mobile-friendly with a modern, classic look. Charts are rendered in scrollable containers for easy navigation on any device.

## Installation

### Prerequisites

- Python 3.7 or above

### Install Dependencies

Install the required packages using pip:

```bash
pip install streamlit yfinance pandas numpy matplotlib scikit-learn mplfinance requests statsmodels
