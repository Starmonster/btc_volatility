# In this file we'll build our web app to investigate bitcoin volatility


import streamlit as st
import datetime
from datetime import timedelta
from plotly.subplots import make_subplots
import talib
import requests
# import matplotlib.pyplot as plt
from volatility_funcs import *
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Set page in wide mode
st.set_page_config(layout="wide")
# Read in the base dataset - exported 17th Aug 2022
btc = pd.read_csv("1D170822tv.csv")
# Clean the data - from function import - format and clean
btc = data_clean(btc)
# Get the latest update - call in latest (post 17-08-22) data
btc_update = get_latest_data(btc)
# Append latest to historical
df = btc.append(btc_update)
# add the volatility feature - percentage difference upper to lower bollinger band 20period/close/2std
df["volatility"] = 1 - (df.lower / df.upper)

# Create a feature to tell the plot what on-hover text to display
hover_text = []
for ind, row in df.iterrows():

    hover_text.append(f"Date:{row.time}<br>"+
                      f"Volatility: {round(row.volatility,3)}<br>"+
                      f"$BTC: ${round(row.close)}"
                     )
df["text"] = hover_text

# Isolate the required columns
df = df[["time", "open", "high", "low", "close",
                         "volume", "basis", "upper", "lower", "volatility", "text"]]

# Note that when plotting the lower bound in plotly there are some infinite / nan log values returned
# in order to avoid this we need to remove any negative lower bound values
c = np.where(df['lower'] < 0 , 0.03, df["lower"])
df["lower"] = c

print(f"before process: {df.index}")



def main(df):

    st.title("BITCOIN VOLATILITY BACKTEST")
    st.text("Back-test price movement subsequent to break of a user-defined volatility threshold")
    st.text("")
    st.sidebar.title("Backtest Parameters")
    # print(df.volatility.mean())

    # Set the user Input
    vol_select = st.sidebar.number_input(label="Enter a volatility threshold between 0.03-0.2", min_value=0.029, max_value=0.199, step=0.001)
    period_select = st.sidebar.number_input(label="Enter a returns period between 20 and 365 days", min_value=20, max_value=365)

    indicator_select = st.sidebar.radio(label="Select a confirmation indicator (beta)", options=["SMA", "EMA", "STOCHRSI", "MACD"])


    # print(f"DF: {df.shape}")

    # Get the plotting data for volatility and timeperiod as specified by user
    plot_vol = volatility_backtest(df=df, vol=vol_select, timeperiod=period_select)

    df.reset_index(inplace=True)

    df = build_indicator(df=df, indicator=indicator_select)

    fig = plot_data(df=df, plot_vol=plot_vol, vol=vol_select, timeperiod=period_select, indicator=indicator_select)

    st.plotly_chart(fig)

    st.subheader("Indicator Parameters")
    st.caption("Volatility = 1 - (lower bollinger band / upper bollinger band) - %age difference lower to upper")
    st.caption("Bollinger Band Parameters: ma=20, std=2, ma-type=sma-close")
    st.caption("MACD Parameters: fast-period=12, slow-period=26, signal=9")
    st.caption("STOCHRSI Parameters: length=14, rsi_length=14, k=3, d=3")




if __name__ == "__main__":
    main(df=df)