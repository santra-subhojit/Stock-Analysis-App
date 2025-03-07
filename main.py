import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.linear_model import LinearRegression
import mplfinance as mpf
import requests
from datetime import datetime, timedelta
import io, base64
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration with wide layout and custom CSS for a classy look
st.set_page_config(page_title="Enhanced Stock Analysis Platform (INR)", layout="wide")
st.markdown("""
    <style>
    /* Sidebar styling */
    .css-1d391kg { font-size: 2.5rem; font-weight: bold; text-align: center; }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #283E51, #4B79A1);
        color: white;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    /* Fix sidebar control panel position */
    .sidebar .sidebar-content .css-1v0mbdj {
        position: sticky;
        top: 0;
    }
    /* Button styling */
    .stButton>button {
        background-color: #4B79A1;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Conversion rate from USD to INR
USD_TO_INR = 75.0

# Expanded ticker list (U.S. and Indian stocks; Indian tickers use ".NS")
SAMPLE_TICKERS = [
    # U.S. Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "BABA", "INTC", "AMD",
    "IBM", "ORCL", "CSCO", "GE", "BA", "F", "GM", "KO", "PEP", "DIS", "WMT", "COST", "MMM", "JNJ",
    # Indian Stocks
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
    "WIPRO.NS", "HCLTECH.NS", "LT.NS", "ULTRACEMCO.NS", "MARUTI.NS", "AXISBANK.NS", "YESBANK.NS",
    "BAJFINANCE.NS", "ZEEL.NS", "HINDUNILVR.NS", "ONGC.NS", "COALINDIA.NS", "TITAN.NS", "DRREDDY.NS",
    "ADANIPORTS.NS", "ADANIGREEN.NS", "SHREECEM.NS", "HDFC.NS", "M&M.NS", "BPCL.NS", "VEDL.NS",
    "NESTLEIND.NS", "IOC.NS", "UPL.NS", "JSWSTEEL.NS", "POWERGRID.NS", "CIPLA.NS", "GRASIM.NS",
    "DIVISLAB.NS", "BIOCON.NS", "SUNPHARMA.NS", "EICHERMOT.NS", "ADANIENT.NS", "PIDILITIND.NS",
    "GODREJCP.NS", "HEROMOTOCO.NS", "COLPAL.NS", "ICICIGI.NS", "SUNTV.NS", "TECHM.NS",
    "MPHASIS.NS", "LUPIN.NS", "JSWENERGY.NS", "BOSCHLTD.NS", "MRPL.NS", "ZOMATO.NS", "NYKAA.NS",
    "IRCTC.NS", "PAYTM.NS", "POLICYBAZAAR.NS", "DELHIVERY.NS"
]

# Alpha Vantage API key (embedded)
ALPHA_API_KEY = "VXT9V6JSOD1JR0IM"

# ---------------------------
# Helper: Scrollable Chart Display
# ---------------------------
def display_scrollable_chart(fig, max_width=1500):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    html = f'<div style="overflow-x: auto; width: 100%;"><img src="data:image/png;base64,{data}" style="max-width: {max_width}px;"/></div>'
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------
# Data Fetching Functions
# ---------------------------
@st.cache_data(show_spinner=False)
def get_stock_data_yf(ticker, start=None, end=None, period='6mo'):
    """Download historical stock data using yfinance and ensure OHLCV columns are numeric."""
    if start and end:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    else:
        data = yf.download(ticker, period=period, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in data.columns:
            if isinstance(data[col], pd.DataFrame):
                data[col] = data[col].iloc[:, 0]
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)
    return data

def get_company_name(ticker):
    """Retrieve the full company name using yfinance."""
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def get_alpha_vantage_daily_data(ticker, start=None, end=None, outputsize="compact", api_key=ALPHA_API_KEY):
    """Fetch daily historical stock data using Alpha Vantage API and filter by date."""
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": ticker, "apikey": api_key, "outputsize": outputsize}
    response = requests.get(url, params=params)
    data = response.json()
    if "Time Series (Daily)" not in data:
        return pd.DataFrame()
    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.apply(pd.to_numeric)
    if start and end:
        mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
        df = df.loc[mask]
    return df

def get_alpha_vantage_global_quote(ticker, api_key=ALPHA_API_KEY):
    """Fetch realtime stock data using Alpha Vantage API."""
    url = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": api_key}
    response = requests.get(url, params=params)
    data = response.json()
    if "Global Quote" not in data:
        return {}
    return data["Global Quote"]

def get_current_status(ticker):
    """Retrieve current day status from yfinance ticker.info."""
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        current = {
            "Open": info.get("regularMarketOpen"),
            "High": info.get("regularMarketDayHigh"),
            "Low": info.get("regularMarketDayLow"),
            "Price": info.get("regularMarketPrice"),
            "Volume": info.get("regularMarketVolume")
        }
        return current
    except Exception:
        return {}

# ---------------------------
# Analysis Functions
# ---------------------------
def calculate_sma(data, window=20):
    """Calculate the SMA on closing prices."""
    return data['Close'].rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal Line."""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_RSI(data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def ml_forecast(data, forecast_days=10, method="Polynomial", degree=2):
    """
    Forecast future closing prices using Polynomial regression (degree-2 by default)
    or ARIMA. Returns a Series of forecasted values.
    """
    if method == "Polynomial":
        df = data.copy().reset_index()
        df['Time'] = np.arange(len(df))
        X = df['Time']
        y = df['Close']
        coeffs = np.polyfit(X, y, degree)
        poly = np.poly1d(coeffs)
        future_times = np.arange(len(df), len(df) + forecast_days)
        forecast_values = poly(future_times)
        forecast_index = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='B')
        return pd.Series(forecast_values, index=forecast_index)
    elif method == "ARIMA":
        try:
            model = ARIMA(data['Close'], order=(1,1,1))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(steps=forecast_days)
            forecast_index = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                           periods=forecast_days, freq='B')
            return pd.Series(forecast_values, index=forecast_index)
        except Exception as e:
            st.error(f"ARIMA forecast error: {e}")
            return None

# ---------------------------
# Plotting Functions using Matplotlib / mplfinance
# ---------------------------
def set_plot_style():
    """Set a clean matplotlib style."""
    available = plt.style.available
    if "seaborn-whitegrid" in available:
        plt.style.use("seaborn-whitegrid")
    else:
        plt.style.use("default")

def plot_line_chart(data, sma, ticker, company_name):
    """Plot a line chart of historical close and SMA."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    data_inr = data['Close'] * USD_TO_INR
    sma_inr = sma * USD_TO_INR
    ax.plot(data.index, data_inr, label="Close Price (INR)", color="blue", linewidth=2,
            marker="o", markersize=4)
    ax.plot(sma.index, sma_inr, label="SMA (20 days) (INR)", color="orange", linestyle='--', linewidth=2)
    ax.set_title(f"Historical Data for {company_name}\n({ticker})", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Price (INR)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)
    max_val = float(data_inr.max())
    min_val = float(data_inr.min())
    max_date = data_inr.idxmax()
    min_date = data_inr.idxmin()
    ax.annotate(f'Max: {max_val:.2f}', xy=(max_date, max_val), 
                xytext=(max_date, max_val * 1.05),
                arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green')
    ax.annotate(f'Min: {min_val:.2f}', xy=(min_date, min_val), 
                xytext=(min_date, min_val * 0.95),
                arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def plot_candlestick_chart(data, ticker, company_name):
    """Plot a candlestick chart with volume and a 20-day moving average using mplfinance."""
    df = data.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    fig, _ = mpf.plot(df, type='candle', volume=True, mav=20,
                      title=f"{company_name} ({ticker}) Candlestick Chart",
                      style="yahoo", returnfig=True)
    return fig

def plot_forecast_chart(data, forecast, ticker, company_name):
    """Plot forecasted prices along with the last 30 days of historical close."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    hist_context = data.tail(30)
    hist_context_inr = hist_context['Close'] * USD_TO_INR
    forecast_inr = forecast * USD_TO_INR
    ax.plot(hist_context.index, hist_context_inr, label="Historical Close (INR)",
            color="blue", linewidth=2, marker="o", markersize=4)
    ax.plot(forecast.index, forecast_inr, label="Forecast (INR)",
            color="green", linestyle='--', marker="o", linewidth=2, markersize=6)
    ax.set_title(f"Forecast for {company_name}\n({ticker})", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Price (INR)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def plot_technical_indicators(data, ticker, company_name):
    """Plot Bollinger Bands, RSI, and MACD in a multi-panel chart."""
    set_plot_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, dpi=100)
    # Bollinger Bands
    price_inr = data['Close'] * USD_TO_INR
    sma_bb, upper_band, lower_band = calculate_bollinger_bands(data)
    sma_inr = sma_bb * USD_TO_INR
    upper_inr = upper_band * USD_TO_INR
    lower_inr = lower_band * USD_TO_INR
    ax1.plot(data.index, price_inr, label="Close Price (INR)", color="blue", linewidth=2)
    ax1.plot(data.index, sma_inr, label="SMA (20)", color="orange", linestyle='--', linewidth=2)
    ax1.fill_between(data.index, lower_inr, upper_inr, color="gray", alpha=0.3, label="Bollinger Bands")
    ax1.set_title(f"Price with Bollinger Bands for {company_name}\n({ticker})", fontsize=16)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # RSI
    rsi = calculate_RSI(data)
    ax2.plot(data.index, rsi, label="RSI (14)", color="purple", linewidth=2)
    ax2.axhline(70, color="red", linestyle="--")
    ax2.axhline(30, color="green", linestyle="--")
    ax2.set_title("RSI", fontsize=14)
    ax2.set_ylabel("RSI", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    # MACD
    macd, signal_line = calculate_macd(data)
    ax3.plot(data.index, macd, label="MACD", color="black", linewidth=2)
    ax3.plot(data.index, signal_line, label="Signal", color="red", linestyle="--", linewidth=2)
    ax3.set_title("MACD", fontsize=14)
    ax3.set_xlabel("Date", fontsize=14)
    ax3.set_ylabel("MACD", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

# ---------------------------
# Main App Layout with Tabs
# ---------------------------
def main():
    st.title("Enhanced Stock & Trading Analysis Dashboard (INR)")
    
    # Sidebar: User controls (sticky control panel)
    st.sidebar.title("Control Panel")
    ticker = st.sidebar.selectbox("Select Ticker Symbol", SAMPLE_TICKERS, index=0)
    chart_type = st.sidebar.radio("Chart Type", ("Line Chart", "Candlestick Chart"))
    forecast_method = st.sidebar.selectbox("Forecast Method", ("Polynomial", "ARIMA"))
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=5, max_value=30, value=10, step=1)
    today = datetime.today().date()
    default_start = today - timedelta(days=182)
    date_range = st.sidebar.date_input("Select Date Range", [default_start, today])
    if len(date_range) != 2:
        st.error("Please select both a start and an end date.")
        return
    start_date, end_date = date_range
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Sidebar: Learn More section about indicators
    with st.sidebar.expander("Learn About Indicators & Forecast Methods"):
        st.markdown("""
        **SMA (Simple Moving Average):**  
        Averages closing prices over a specified period to smooth out price fluctuations.
        
        **Candlestick Chart:**  
        Shows open, high, low, and close prices to provide insight into market sentiment.
        
        **Bollinger Bands:**  
        Volatility bands placed above and below the SMA indicating overbought/oversold conditions.
        
        **RSI (Relative Strength Index):**  
        A momentum oscillator that measures price changes to identify overbought/oversold conditions.
        
        **MACD (Moving Average Convergence Divergence):**  
        Shows the relationship between two moving averages to highlight momentum shifts.
        
        **Polynomial Forecast:**  
        Uses polynomial regression to capture non-linear trends in price data.
        
        **ARIMA Forecast:**  
        A time series forecasting model that uses autoregressive, integrated, and moving average components.
        """)
    
    # Create tabs for separate analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Historical Chart", "Technical Analysis", "Forecast & Predictions", "Advanced Details"])
    
    if st.sidebar.button("Fetch Data"):
        with st.spinner(f"Fetching data for {ticker} from {start_str} to {end_str}..."):
            data = get_stock_data_yf(ticker, start=start_str, end=end_str)
            source = "yfinance"
            if data.empty:
                data = get_alpha_vantage_daily_data(ticker, outputsize="compact", api_key=ALPHA_API_KEY)
                source = "Alpha Vantage"
            if data.empty:
                st.error(f"No historical data returned for {ticker}. Try a different ticker or date range.")
                return
            company_name = get_company_name(ticker) if source=="yfinance" else ticker
            sma = calculate_sma(data)
            if forecast_method == "ARIMA":
                try:
                    model = ARIMA(data['Close'], order=(1,1,1))
                    model_fit = model.fit()
                    forecast_values = model_fit.forecast(steps=forecast_days)
                    forecast = pd.Series(forecast_values, index=pd.date_range(start=data.index[-1] + timedelta(days=1),
                                                                               periods=forecast_days, freq='B'))
                except Exception as e:
                    st.error(f"ARIMA forecast error: {e}")
                    return
            else:
                forecast = ml_forecast(data, forecast_days=forecast_days, degree=2)
            current_status = get_current_status(ticker)
            
            st.success(f"Data updated for {company_name} ({ticker}) using {source}.")
            
            with tab1:
                st.subheader("Current Day Status")
                if current_status:
                    st.table(pd.DataFrame(current_status, index=[0]))
                else:
                    st.write("No current status available.")
                st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            with tab2:
                st.subheader(f"Historical Data for {company_name} ({ticker})")
                if chart_type == "Line Chart":
                    fig_hist = plot_line_chart(data, sma, ticker, company_name)
                else:
                    fig_hist = plot_candlestick_chart(data, ticker, company_name)
                display_scrollable_chart(fig_hist)
            
            with tab3:
                st.subheader(f"Technical Analysis for {company_name} ({ticker})")
                fig_tech = plot_technical_indicators(data, ticker, company_name)
                display_scrollable_chart(fig_tech)
            
            with tab4:
                st.subheader(f"Forecast for {company_name} ({ticker})")
                fig_forecast = plot_forecast_chart(data, forecast, ticker, company_name)
                display_scrollable_chart(fig_forecast)
                st.subheader("Forecast Details")
                forecast_df = forecast.to_frame(name="Predicted Close (INR)")
                forecast_df.index.name = "Date"
                st.table(forecast_df)
                
            with tab5:
                st.subheader("Advanced Analysis Details")
                st.markdown("Additional analysis such as correlation metrics, trading signals, or multiple forecasting models can be added here.")
            
            if source == "Alpha Vantage":
                quote = get_alpha_vantage_global_quote(ticker, api_key=ALPHA_API_KEY)
                if quote:
                    st.subheader("Realtime Data (Alpha Vantage)")
                    st.json(quote)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This platform provides an innovative stock analysis experience with detailed historical charts, technical indicators, "
        "and advanced forecasting models. Use the controls above to adjust parameters, and refer to the 'Learn About' section for "
        "explanations of each indicator and forecast method."
    )

if __name__ == "__main__":
    main()
