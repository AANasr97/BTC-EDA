import pandas as pd
from datetime import datetime

def preprocess_and_resample(data_file):
    # Read the data from the CSV file
    BTC = pd.read_csv(data_file)

    # Convert milliseconds to seconds by dividing by 1000 and apply to the 'time' column
    BTC['time'] = BTC['time'] / 1000

    # Convert to a human-readable date and time format
    BTC['time'] = BTC['time'].apply(lambda x: datetime.utcfromtimestamp(x))

    # Set the 'time' column as the index of the DataFrame
    BTC.set_index('time', inplace=True)

    # Resample data at different intervals
    intervals = ['1T', '15T', '30T', '1H', '4H', 'D', 'W', 'M', 'Q']
    resampled_dataframes = {}

    for interval in intervals:
        resampled_dataframes[interval] = BTC.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Calculate the percentage change for the 'close' column
        resampled_dataframes[interval]['Return'] = resampled_dataframes[interval]['close'].pct_change()

        # Drop the NaN value in the 'Return' column (first row)
        resampled_dataframes[interval] = resampled_dataframes[interval].dropna()

        # Calculate 'Total Traded' column
        resampled_dataframes[interval]['Total Traded'] = (
            resampled_dataframes[interval]['close'] * resampled_dataframes[interval]['volume']
        )

    # Unpack the resampled DataFrames into separate variables
    BTC_1min = resampled_dataframes['1T']
    BTC_15min = resampled_dataframes['15T']
    BTC_30min = resampled_dataframes['30T']
    BTC_1hr = resampled_dataframes['1H']
    BTC_4hr = resampled_dataframes['4H']
    BTC_daily = resampled_dataframes['D']
    BTC_weekly = resampled_dataframes['W']
    BTC_monthly = resampled_dataframes['M']
    BTC_quarterly = resampled_dataframes['Q']

    return (
        BTC_1min, BTC_15min, BTC_30min, BTC_1hr,
        BTC_4hr, BTC_daily, BTC_weekly, BTC_monthly, BTC_quarterly
    )

# Example usage:
data_file = 'btcusd.csv'
(
    BTC_1min, BTC_15min, BTC_30min, BTC_1hr,
    BTC_4hr, BTC_daily, BTC_weekly, BTC_monthly, BTC_quarterly
) = preprocess_and_resample(data_file)



import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def summary_ann(returns):
    summary = returns.agg(["mean", "std"]).T
    summary["Return"] = summary["mean"] * 365
    summary["Risk"] = summary["std"] * np.sqrt(365)
    summary.drop(columns=["mean", "std"], inplace=True)
    return summary

def execute_strategy(df, strategy_name):
    st.write(f'You selected the {strategy_name} strategy.')

    def simple_momentum_strategy(df):
        df['Positions'] = np.sign(df['Return'])
        df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
        df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()
        df.dropna(inplace=True)
        return df

    def simple_contrarian_strategy(df):
        df['Positions'] = -np.sign(df['Return'])
        df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
        df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()
        df.dropna(inplace=True)
        return df

    def threshold_based_strategy(df):
        df['Positions'] = np.where(df['Return'] > 0.3, -1, 1)
        df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
        df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()
        df.dropna(inplace=True)
        return df

    def sma1_strategy(df):
        df["SMA50"] = df.close.rolling(window=50).mean()
        df["SMA200"] = df.close.rolling(window=200).mean()
        df['Positions'] = np.sign(df.SMA50.sub(df.SMA200))
        df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
        df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()
        df.dropna(inplace=True)
        return df

    def sma2_strategy(df):
        df["SMA50"] = df.close.rolling(window=50).mean()
        df["SMA200"] = df.close.rolling(window=200).mean()
        df['Positions'] = -np.sign(df.SMA50.sub(df.SMA200))
        df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
        df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()
        df.dropna(inplace=True)
        return df

    def ewma_strategy(df):
        df['EWMA12'] = df['close'].ewm(span=12).mean()
        df['Positions'] = np.where(df['close'] > df['EWMA12'], 1, -1)
        df["Strategy_Ret"] = df["Positions"].shift() * df["Return"]
        df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()
        df.dropna(inplace=True)
        return df

    def perfect_strategy(df):
        df['Positions'] = np.sign(df['Return'])
        df["Strategy_Ret"] = df["Positions"] * df["Return"]
        df["Strategy"] = (1 + df["Strategy_Ret"]).cumprod()
        df.dropna(inplace=True)
        return df

    strategies = {
        'Simple Momentum': simple_momentum_strategy,
        'Simple Contrarian': simple_contrarian_strategy,
        'Threshold_based': threshold_based_strategy,
        'SMA1': sma1_strategy,
        'SMA2': sma2_strategy,
        'EWMA': ewma_strategy,
        'Perfect': perfect_strategy,
    }

    if strategy_name not in strategies:
        st.write("Strategy not implemented.")
        return

    strategy_function = strategies[strategy_name]
    strategy_data = strategy_function(df)

    # Create a Plotly figure for the strategy and asset's closing price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['close'], mode='lines', name='Asset Close Price'))
    fig.add_trace(go.Scatter(x=strategy_data.index, y=strategy_data['Strategy'], mode='lines', name='Strategy Performance'))
    fig.update_layout(
        title=f'{strategy_name} Strategy',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    fig.update_layout(
        autosize=True,
        width=1200,
        height=800
    )

    st.plotly_chart(fig)

    # Display the strategy data and summary
    st.dataframe(strategy_data.head(3))
    st.dataframe(strategy_data.tail(3))
    st.dataframe(summary_ann(strategy_data[["Return", "Strategy_Ret"]]))

# Load your data into EDA_data
EDA_data = pd.read_csv('btcusd.csv')
EDA_data['time'] = pd.to_datetime(EDA_data['time'], unit='s')
EDA_data.set_index('time', inplace=True)

# Select strategy
strategy = st.selectbox('Select Strategy:', ['Simple Momentum', 'Simple Contrarian', 'Threshold_based', 'SMA1', 'SMA2', 'EWMA', 'Perfect'])

# Execute selected strategy
execute_strategy(EDA_data, strategy)
