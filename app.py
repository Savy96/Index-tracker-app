import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go

# --- Metric Calculation Functions ---
def calculate_cagr(start_value, end_value, years):
    if start_value > 0 and years > 0:
        cagr = (end_value / start_value) ** (1 / years) - 1
        return round(cagr * 100, 2)  # Convert to percentage and round
    return np.nan

def calculate_drawdown(current_price, ath_price):
    return round(((current_price - ath_price) / ath_price) * 100, 2)

def calculate_annualized_volatility(df):
    daily_returns = df['Close'].pct_change().dropna()
    return round(daily_returns.std() * np.sqrt(252), 2)

def calculate_sharpe_ratio(df, risk_free_rate=0.0):
    daily_returns = df['Close'].pct_change().dropna()
    excess_returns = daily_returns - risk_free_rate / 252
    return round((excess_returns.mean() / excess_returns.std()) * np.sqrt(252), 2)

def calculate_sortino_ratio(df, risk_free_rate=0.0):
    daily_returns = df['Close'].pct_change().dropna()
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std()
    if downside_std == 0:
        return np.nan
    excess_returns = daily_returns - risk_free_rate / 252
    return round((excess_returns.mean() / downside_std) * np.sqrt(252), 2)

def calculate_max_drawdown_period(df):
    prices = df['Close']
    roll_max = prices.cummax()
    drawdowns = (prices - roll_max) / roll_max
    min_drawdown = drawdowns.min()
    end = drawdowns.idxmin()
    start = prices[:end].idxmax()
    return start, end, round(min_drawdown * 100, 2)

def filter_timeframe(df, option, start_date=None, end_date=None):
    if option == "Custom" and start_date and end_date:
        return df[(df.index >= start_date) & (df.index <= end_date)]
    if option == "1M":
        return df[df.index >= datetime.today() - timedelta(days=30)]
    elif option == "6M":
        return df[df.index >= datetime.today() - timedelta(days=182)]
    elif option == "YTD":
        return df[df.index >= datetime(datetime.today().year, 1, 1)]
    elif option == "1Y":
        return df[df.index >= datetime.today() - timedelta(days=365)]
    elif option == "5Y":
        return df[df.index >= datetime.today() - timedelta(days=5*365)]
    else:
        return df

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

# Mapping of index names to Yahoo Finance tickers
ticker_map = {
    'Nifty 50': '^NSEI',
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'Nasdaq': '^IXIC',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'Nikkei 225': '^N225',
    'Hang Seng': '^HSI',
    'Shanghai Composite': '000001.SS',
    'Euro Stoxx 50': '^STOXX50E',
    'Straits Times': '^STI',
    'MSCI World': 'URTH',
    'MSCI Emerging Markets': 'EEM',
    'MSCI Developed Markets': 'VEA'
}

st.set_page_config(page_title="Index Tracker", layout="wide")
st.title("\U0001F4C8 Global Index Tracker")
st.markdown("Compare index performance and see CAGR based on live data from Yahoo Finance.")

# User date input via date_input
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2015, 3, 31))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.min.time())
years = (end_date - start_date).days / 365.25

selected_indices = st.multiselect("Select indices to track:", options=["None"] + list(ticker_map.keys()))
if "None" in selected_indices:
    selected_indices = []

# Fetch historical prices with loading spinner
data = []
historical_data = {}
alerts = []
last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with st.spinner("Fetching data from Yahoo Finance..."):
    for name in selected_indices:
        ticker = ticker_map[name]
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is None or df.empty:
                st.warning(f"No data returned for {name} ({ticker})")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna(subset=['Close'])
            if df.empty:
                st.warning(f"No usable 'Close' data for {name} ({ticker})")
                continue

            start_val = round(df['Close'].iloc[0], 2)
            end_val = round(df['Close'].iloc[-1], 2)
            cagr = calculate_cagr(start_val, end_val, years)
            price_return = ((end_val - start_val) / start_val) * 100
            ath_val = round(df['Close'].max(), 2)
            ath_date = df['Close'].idxmax().strftime('%Y-%m-%d')
            drawdown = calculate_drawdown(end_val, ath_val)
            high_52w = round(df['Close'][-252:].max(), 2)
            low_52w = round(df['Close'][-252:].min(), 2)
            volatility = calculate_annualized_volatility(df)
            sharpe_ratio = calculate_sharpe_ratio(df)
            sortino_ratio = calculate_sortino_ratio(df)
            dd_start = None
            dd_end = None
            dd_min = None
            if len(df) > 0:
              dd_start, dd_end, dd_min = calculate_max_drawdown_period(df)

            if end_val >= high_52w:
                alerts.append(f"\U0001F680 {name} is at a new 52-week HIGH!")
            if end_val <= low_52w:
                alerts.append(f"\U0001F53B {name} is at a new 52-week LOW!")

            data.append({
                'Index': name,
                'Start Value': start_val,
                'End Value': end_val,
                'CAGR': cagr,
                'ATH Value': ath_val,
                'ATH Date': ath_date,
                'Drawdown from ATH (%)': drawdown,
                'Max Drawdown (%)': dd_min,
                'Max Drawdown Start': dd_start.strftime('%Y-%m-%d'),
                'Max Drawdown End': dd_end.strftime('%Y-%m-%d'),
                '52W High': high_52w,
                '52W Low': low_52w,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio
            })

            df['Index'] = name
            historical_data[name] = df

        except Exception as e:
            st.error(f"Error loading {name} ({ticker}): {e}")

# Display table and alerts
if data:
    st.caption(f"Last updated: {last_updated}")

    if alerts:
        for alert in alerts:
            st.warning(alert)

    df = pd.DataFrame(data)

    # Format CAGR column as percentage
    df['CAGR'] = df['CAGR'].astype(float).map("{:.2f}%".format)
    df['Drawdown from ATH (%)'] = df['Drawdown from ATH (%)'].astype(float).map("{:.2f}%".format)
    df['Max Drawdown (%)'] = df['Max Drawdown (%)'].astype(float).map("{:.2f}%".format)
    df['52W High'] = df['52W High'].astype(float).map("{:.2f}".format)
    df['52W Low'] = df['52W Low'].astype(float).map("{:.2f}".format)
    df['Volatility'] = df['Volatility'].astype(float).map("{:.2f}".format)
    df['Sharpe Ratio'] = df['Sharpe Ratio'].astype(float).map("{:.2f}".format)
    df['Sortino Ratio'] = df['Sortino Ratio'].astype(float).map("{:.2f}".format)

    st.dataframe(df, use_container_width=True)

    st.subheader("\U0001F4CA CAGR Comparison")
    df_bar = df[['Index', 'CAGR']].copy()
    fig_cagr = px.bar(df_bar, x='Index', y='CAGR', color='Index')
    st.plotly_chart(fig_cagr, use_container_width=True)

    st.subheader("\U0001F4C9 Correlation Heatmap")
    daily_returns = pd.DataFrame({name: historical_data[name]['Close'].pct_change() for name in historical_data})
    corr = daily_returns.corr().round(2)
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.round(2).astype(str).values,
        colorscale='RdBu',
        showscale=True,
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("\U0001F4C6 Calendar Year Returns")
    yearly_returns = pd.DataFrame()
    for name, df in historical_data.items():
        df['Year'] = df.index.year
        yearly = df.groupby('Year')['Close'].apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
        yearly_returns[name] = yearly

    yearly_returns_plot = yearly_returns.fillna(0).sort_index(ascending=False)
    fig_calendar = px.imshow(
        yearly_returns_plot.values,
        labels=dict(x="Index", y="Year", color="Return (%)"),
        x=yearly_returns_plot.columns,
        y=yearly_returns_plot.index,
        color_continuous_scale=["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
        text_auto=True
    )
    fig_calendar.update_layout(title="Calendar Year Returns Heatmap", yaxis_nticks=len(yearly_returns_plot.index))
    st.plotly_chart(fig_calendar, use_container_width=True)

    st.subheader("\U0001F4C8 Price Over Time")
    st.markdown("### Chart Options")
    col_tf, col_ma, col_norm = st.columns([1, 2, 1])
    with col_tf:
        timeframe_option = st.radio("Select Timeframe", ["1M", "6M", "YTD", "1Y", "5Y", "Max", "Custom"], horizontal=True)
    with col_ma:
        ma_enabled = st.checkbox("Overlay Moving Averages")
        ma_periods = []
        if ma_enabled:
            ma_input = st.text_input("Enter MA Periods (e.g., 50,200)", "50,200")
            ma_periods = [int(x.strip()) for x in ma_input.split(',') if x.strip().isdigit()]
    with col_norm:
        normalize_prices = st.checkbox("Normalize Prices (Start at 100)")

    for name, df in historical_data.items():
        for ma in ma_periods:
            df[f"MA{ma}"] = df['Close'].rolling(window=ma).mean()

    combined_df = pd.concat(historical_data.values())

    # Apply both timeframe and date filtering
    filtered_df = filter_timeframe(combined_df, timeframe_option)
    filtered_df = filtered_df[(filtered_df.index >= start_date) & (filtered_df.index <= end_date)]

    fig_price = px.line()
    for index in filtered_df['Index'].unique():
        df_index = filtered_df[filtered_df['Index'] == index]
        y_data = (df_index['Close'] / df_index['Close'].iloc[0] * 100) if normalize_prices else df_index['Close']
        fig_price.add_scatter(x=df_index.index, y=y_data, mode='lines', name=index)
        if ma_enabled:
            for ma in ma_periods:
                if f"MA{ma}" in df_index.columns:
                    fig_price.add_scatter(
                        x=df_index.index,
                        y=(df_index[f"MA{ma}"] / df_index['Close'].iloc[0] * 100) if normalize_prices else df_index[f"MA{ma}"],
                        mode='lines',
                        name=f"{index} MA{ma}",
                        line=dict(dash='dot')
                    )
    fig_price.update_layout(title='Price Performance')
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("\U0001F680 Technical Analysis")
    
    if selected_indices:
        # Define RSI and MACD functions here (before the loop)
        def calculate_rsi(df, window=14):
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        def calculate_macd(df):
            short_ema = df['Close'].ewm(span=12, adjust=False).mean()
            long_ema = df['Close'].ewm(span=26, adjust=False).mean()
            macd = short_ema - long_ema
            signal_line = macd.ewm(span=9, adjust=False).mean()
            return macd, signal_line

        for name, df in historical_data.items():

            # Timeframe selection
            tf_option = st.radio(
                f"Select Timeframe for Technical Analysis for {name}:",
                ["1M", "6M", "YTD", "1Y", "5Y", "Max", "Custom"],
                key=f"tf_option_{name}",  # Unique key for each radio
                horizontal=True
            )

            # Apply timeframe filtering
            df_filtered = filter_timeframe(df.copy(), tf_option, start_date, end_date)

            # Ensure DatetimeIndex and handle missing values
            if not isinstance(df_filtered.index, pd.DatetimeIndex):
                df_filtered.index = pd.to_datetime(df_filtered.index)

            # Reindex to business days
            all_days = pd.date_range(start=df_filtered.index.min(), end=df_filtered.index.max(), freq='D')
            business_days = all_days[all_days.weekday < 5]
            df_filtered = df_filtered.reindex(business_days)
            df_filtered.ffill(inplace=True)
            df_filtered.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

            # Calculate technical indicators
            df_filtered['MA20'] = df_filtered['Close'].rolling(window=20).mean()
            df_filtered['Upper_Band'] = df_filtered['MA20'] + (df_filtered['Close'].rolling(window=20).std() * 2)
            df_filtered['Lower_Band'] = df_filtered['MA20'] - (df_filtered['Close'].rolling(window=20).std() * 2)
            df_filtered['RSI'] = calculate_rsi(df_filtered)
            df_filtered['MACD'], df_filtered['Signal_Line'] = calculate_macd(df_filtered)

            # Indicator selection
            indicator_option = st.radio(
                f"Select indicator to overlay on Candlestick Chart for {name}:",
                ["None", "Bollinger Bands", "RSI", "MACD"],
                key=f"indicator_option_{name}",  # Unique key for each radio
                horizontal=True
            )

            # Create candlestick chart
            fig_candlestick = go.Figure()

            if indicator_option == "Bollinger Bands":
                fig_candlestick = go.Figure(data=[go.Candlestick(
                    x=df_filtered.index,
                    open=df_filtered['Open'],
                    high=df_filtered['High'],
                    low=df_filtered['Low'],
                    close=df_filtered['Close'],
                    name='Candlesticks'
                )])

                fig_candlestick.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Upper_Band'], mode='lines', name='Upper Band',
                                                     line=dict(color='purple')))
                fig_candlestick.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Lower_Band'], mode='lines', name='Lower Band',
                                                     line=dict(color='purple')))

                fig_candlestick.update_layout(
                    title=f'Candlestick Chart with Bollinger Bands for {name}',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_candlestick, use_container_width=True)

            # Add RSI
            elif indicator_option == "RSI":
                # Create a new subplot for RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['RSI'], mode='lines', name='RSI', line=dict(color='blue')))
                fig_rsi.add_hline(y=70, line_dash="dot", annotation_text="Overbought", annotation_position="top left")
                fig_rsi.add_hline(y=30, line_dash="dot", annotation_text="Oversold", annotation_position="bottom left")
                fig_rsi.update_layout(title=f'RSI for {name}', xaxis_title='Date', yaxis_title='RSI')
                st.plotly_chart(fig_rsi, use_container_width=True)

            # Add MACD
            elif indicator_option == "MACD":
                # Create a new subplot for MACD
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MACD'], mode='lines', name='MACD', line=dict(color='green')))
                fig_macd.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Signal_Line'], mode='lines', name='Signal Line',
                                                 line=dict(color='red')))
                fig_macd.update_layout(title=f'MACD for {name}', xaxis_title='Date', yaxis_title='MACD')
                st.plotly_chart(fig_macd, use_container_width=True)

            # Display Candlestick Chart without overlay
            else:
                # Add candlestick chart
                fig_candlestick = go.Figure(data=[go.Candlestick(
                    x=df_filtered.index,
                    open=df_filtered['Open'],
                    high=df_filtered['High'],
                    low=df_filtered['Low'],
                    close=df_filtered['Close'],
                    name='Candlesticks'
                )])
                fig_candlestick.update_layout(
                    title=f'Candlestick Chart for {name}',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_candlestick, use_container_width=True)

else:
    st.info("No index data available to display.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + yfinance")
