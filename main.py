import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import mplfinance as mpf
import requests
from datetime import datetime, timedelta
import io, base64
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration and custom CSS for a modern, clean, and responsive look
st.set_page_config(page_title="QuantiQ (Stock Analysis Platform)", layout="wide")
st.markdown("""
    <style>
    /* Global background and fonts */
    .reportview-container {
        background: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1f2a44, #3e4a70);
        color: #f0f0f0;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2 {
        color: #f0f0f0;
    }
    .sidebar .sidebar-content .css-1v0mbdj {
        position: sticky;
        top: 0;
    }
    /* Button styling */
    .stButton>button {
        background-color: #3e4a70;
        color: #fff;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 10px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1f2a44;
    }
    /* Responsive header with two spans */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
    }
    .full-title { display: inline; }
    .short-title { display: none; }
    /* Control panel arrow hint for small screens */
    .control-arrow {
        display: block;
        position: fixed;
        bottom: 10px;
        left: 10px;
        font-size: 16px;
        color: #1f2a44;
        animation: fadeInOut 2s infinite;
        z-index: 1000;
    }
    @keyframes fadeInOut {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
    @media only screen and (max-width: 600px) {
        .main-title { font-size: 1.8rem; }
        .full-title { display: none; }
        .short-title { display: inline; }
        .stButton>button { font-size: 14px; padding: 8px 16px; }
        .reportview-container { padding: 10px; }
    }
    </style>
    """, unsafe_allow_html=True)

# Responsive Header: Two spans for full and short title text
st.markdown('''
    <h1 class="main-title">
       <span class="full-title">QuantiQ (Stock Analysis Platform)</span>
       <span class="short-title">QuantiQ</span>
    </h1>
''', unsafe_allow_html=True)

# Conversion rate from USD to INR
USD_TO_INR = 75.0

# Expanded and updated ticker list (both US and Indian stocks)
# These lists include additional in-demand stocks and span multiple sectors.
SAMPLE_TICKERS = [
    # US Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "AMD", "IBM", "ORCL", "CSCO",
    "DIS", "KO", "PEP", "WMT", "COST", "SQ", "UBER", "ZM", "SNOW",
    # Indian Stocks
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "MPHASIS.NS",
    "DRREDDY.NS", "CIPLA.NS", "BIOCON.NS", "SUNPHARMA.NS", "LUPIN.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS", "M&M.NS",
    "AXISBANK.NS", "YESBANK.NS", "BAJFINANCE.NS", "ICICIPRULI.NS",
    "BHARTIARTL.NS", "NESTLEIND.NS",
    "ONGC.NS", "COALINDIA.NS", "ADANIPORTS.NS", "ADANIGREEN.NS",
    "JSWSTEEL.NS", "POWERGRID.NS", "SHREECEM.NS", "COLPAL.NS", "BOSCHLTD.NS",
    # Additional high-growth / trending Indian stocks:
    "PIDILITIND.NS", "TITAN.NS", "COFORGE.NS", "UPL.NS", "VEDL.NS", "GODREJCP.NS"
]

# Define diverse stock categories with more in-demand and sector-specific stocks
stock_categories = {
    "All": SAMPLE_TICKERS,
    "Tech": [
        # US Tech stocks
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "AMD", "IBM", "ORCL", "CSCO", "SQ", "ZM", "SNOW",
        # Indian IT/Tech stocks
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "MPHASIS.NS", "COFORGE.NS"
    ],
    "Healthcare": [
        "DRREDDY.NS", "CIPLA.NS", "BIOCON.NS", "SUNPHARMA.NS", "LUPIN.NS",
        # Optionally, add a major US healthcare name if desired, e.g. "JNJ" (if you wish to extend US exposure)
        "JNJ"
    ],
    "Automobile": [
        "MARUTI.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS", "M&M.NS"
    ],
    "Finance": [
        "ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS", "AXISBANK.NS", "YESBANK.NS", "BAJFINANCE.NS", "ICICIPRULI.NS"
    ],
    "Consumer": [
        "RELIANCE.NS", "BHARTIARTL.NS", "NESTLEIND.NS", "WMT", "COST", "PEP", "GODREJCP.NS", "TITAN.NS", "PIDILITIND.NS"
    ],
    "Energy": [
        "ONGC.NS", "COALINDIA.NS", "ADANIPORTS.NS", "ADANIGREEN.NS"
    ],
    "Industrial": [
        "JSWSTEEL.NS", "POWERGRID.NS", "SHREECEM.NS", "COLPAL.NS", "BOSCHLTD.NS", "UPL.NS", "VEDL.NS"
    ]
}


# Sidebar: Control Panel with Category and Stocks selectboxes
st.sidebar.title("Control Panel")
selected_category = st.sidebar.selectbox("Category", list(stock_categories.keys()))
available_tickers = stock_categories[selected_category]
ticker = st.sidebar.selectbox("Stocks", available_tickers, index=0)
chart_type = st.sidebar.radio("Chart Type", ("Line Chart", "Candlestick Chart"))
forecast_method = st.sidebar.selectbox("Forecast Method", ("Polynomial", "ARIMA"))
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=5, max_value=30, value=10, step=1)
today = datetime.today().date()
default_start = today - timedelta(days=182)
date_range = st.sidebar.date_input("Date Range", [default_start, today])
if len(date_range) != 2:
    st.error("Select both start and end dates.")
    st.stop()
start_date, end_date = date_range
start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

# Add control panel arrow hint for small screens
st.sidebar.markdown('<div class="control-arrow">&#8592; Tap on the control panel for fetching data</div>', unsafe_allow_html=True)

# Initialize session state for data fetching
if "data_fetched" not in st.session_state:
    st.session_state.data_fetched = False

# The Results button (formerly "Fetch Data")
if st.sidebar.button("Results"):
    st.session_state.data_fetched = True

# Create tabs (About is always visible)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Current Status", "Historical Chart", "Technical Analysis", "Forecast & Predictions", "About"])

# ---------------------------
# Helper: Scrollable Chart Display (Responsive)
# ---------------------------
def display_scrollable_chart(fig):
    """Convert Matplotlib figure to responsive HTML image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    html = f'<div style="overflow-x: auto; width: 100%;"><img src="data:image/png;base64,{data}" style="width: 100%; max-width: 1500px;"/></div>'
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------
# Data Fetching Functions
# ---------------------------
@st.cache_data(show_spinner=False)
def get_stock_data_yf(ticker, start=None, end=None, period='6mo'):
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
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def get_alpha_vantage_daily_data(ticker, start=None, end=None, outputsize="compact", api_key="VXT9V6JSOD1JR0IM"):
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

def get_alpha_vantage_global_quote(ticker, api_key="VXT9V6JSOD1JR0IM"):
    url = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": api_key}
    response = requests.get(url, params=params)
    data = response.json()
    if "Global Quote" not in data:
        return {}
    return data["Global Quote"]

def get_current_status(ticker):
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
    return data['Close'].rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def ml_forecast(data, forecast_days=10, method="Polynomial", degree=2):
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
    available = plt.style.available
    if "seaborn-whitegrid" in available:
        plt.style.use("seaborn-whitegrid")
    else:
        plt.style.use("default")

def plot_line_chart(data, sma, ticker, company_name):
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
# Main App Layout and Tab Content
# ---------------------------
def main():
    ALPHA_API_KEY = "VXT9V6JSOD1JR0IM"
    
    if st.session_state.data_fetched:
        with st.spinner(f"Fetching data for {ticker} from {start_str} to {end_str}..."):
            data = get_stock_data_yf(ticker, start=start_str, end=end_str)
            source = "yfinance"
            if data.empty:
                data = get_alpha_vantage_daily_data(ticker, outputsize="compact", api_key=ALPHA_API_KEY)
                source = "Alpha Vantage"
            if data.empty:
                st.error(f"No historical data for {ticker}. Try another ticker or date range.")
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
            
            # Optionally display realtime data if using Alpha Vantage
            if source == "Alpha Vantage":
                quote = get_alpha_vantage_global_quote(ticker, api_key=ALPHA_API_KEY)
                if quote:
                    st.subheader("Realtime Data (Alpha Vantage)")
                    st.json(quote)
    else:
        for tab in (tab1, tab2, tab3, tab4):
            with tab:
                st.info("Tap 'Results' in the Control Panel to fetch data for analysis.")
                
    # About tab is always visible
    with tab5:
        st.subheader("About This Platform")
        st.markdown("""
        **QuantiQ (Stock Analysis Platform)** provides a comprehensive suite of trading analysis tools:
        
        - **Current Status:** Displays the current market status and realtime updates.
        - **Historical Chart:** Visualize historical price data as a line or candlestick chart with volume and a 20-day moving average.
        - **Technical Analysis:** Explore key technical indicators such as Bollinger Bands, RSI, and MACD.
        - **Forecast & Predictions:** Forecast future prices using advanced Polynomial or ARIMA models.
        
        **Glossary:**
        - **SMA:** Simple Moving Average – smooths price data over time.
        - **Candlestick Chart:** Visualizes open, high, low, and close prices to reveal market sentiment.
        - **Bollinger Bands:** Volatility bands that indicate overbought or oversold conditions.
        - **RSI:** Relative Strength Index – a momentum oscillator indicating overbought/oversold levels.
        - **MACD:** Moving Average Convergence Divergence – measures momentum changes.
        - **Polynomial Forecast:** Uses polynomial regression for capturing non-linear trends.
        - **ARIMA Forecast:** A statistical forecasting method using autoregressive integrated moving averages.
        
        Use the Control Panel to adjust parameters and explore the data.
        """)
        
    st.sidebar.markdown("---")
    st.sidebar.info(
        "QuantiQ provides an innovative, responsive, and trading-friendly analysis experience with detailed charts, "
        "technical indicators, and forecasting tools. Use the Control Panel to customize your analysis."
    )

if __name__ == "__main__":
    main()
