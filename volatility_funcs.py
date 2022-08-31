# Functions associated with btc volatility web app


import pandas as pd
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly.express as px
import talib
import pandas_ta as ta
import requests
import warnings
warnings.filterwarnings("ignore")


# Function to clean the read in historical data
def data_clean(btc):
    # Base clean of data - isolate columns, get datetime format, remove nans etc
    cols = ['time', 'open', 'high', 'low', 'close', 'Volume', 'Basis', 'Upper', 'Lower']
    btc = btc[cols]
    btc.time = pd.to_datetime(btc["time"], unit='s')
    btc.dropna(axis=0, inplace=True)
    # Change feature headers to suit standard
    btc.rename(columns={"Volume": "volume", "Basis": "basis", "Upper": "upper", "Lower": "lower"}, inplace=True)

    return btc

def get_latest_data(btc):
    #### USE COINBASE API TO CALL IN LATEST DATA

    # Get the url and params
    apiUrl = "https://api.pro.coinbase.com"
    sym = "BTC-USD"
    # Set the granularity - we want daily so 86400 seconds per day
    barSize = "86400"
    timeEnd = datetime.datetime.now()
    delta = timedelta(days=100)
    timeStart = timeEnd - (1 * delta)

    parameters = {
        "start": timeStart,
        "end": timeEnd,
        "granularity": barSize

    }

    # CALL IN THE DATA
    data = requests.get(f"{apiUrl}/products/{sym}/candles",
                        params=parameters,
                        headers={"content-type": "application/json"}

                        )

    # SET INTO DF AND Reformat time
    btc_cbs = pd.DataFrame(data.json(), columns=["time", "low", "high", "open", "close", "volume"])

    btc_cbs["time"] = pd.to_datetime(btc_cbs["time"], unit="s")

    # Invert ( to most recent at tail)
    btc_latest = btc_cbs.iloc[::-1]
    btc_latest.reset_index(inplace=True, drop=True)
    btc_latest.tail()

    # Buil latest bbs
    bollies = talib.BBANDS(btc_latest["close"], timeperiod=20, nbdevup=2, nbdevdn=2)


    bollies = pd.DataFrame(bollies).T.rename(columns={0: "upper", 1: "basis", 2: "lower"})

    # APPEND bollinger band data to latest btc read in
    btc_bb = btc_latest.join(bollies)

    # Now isolate only the dates that we need updates for to speed up the code
    btc_update = btc_bb[btc_bb.time.between(btc.iloc[-1].time, btc_bb.iloc[-2].time)].iloc[1:]


    return btc_update



# Run the volatility backtest
def volatility_backtest(df, vol=0.2, timeperiod=100):
    data_cols = ["thresh_date", "price", "gain", "fall", "volatility", "period"]

    vol_data = pd.DataFrame(columns=data_cols)

    # vol = float(input("What volatility level to check?"))
    #
    # timeperiod = input("Over what time period to check performance")

    # Get the dataframe of all volatility values less than the user input
    vol_df = df[df.volatility < vol]
    # print(f"VOL DF COLS: {vol_df.columns, vol_df.shape}")


    # vol_df["month"] = vol_df.index.month
    # vol_df["year"] = vol_df.index.year
    # vol_df["month"] = vol_df.time.month
    # vol_df["year"] = vol_df.time.year

    unq = []
    for ind, row in vol_df.iterrows():
        # month = ind.month
        # year = ind.year
        month = row.time.month
        year = row.time.year

        unq.append((year, month))
    vol_df["unique"] = unq
    # print(f"UNIQUE: {unq}")
    list_of_dates = vol_df.unique.unique()
    # print(f"LIST {list_of_dates[:6], len(list_of_dates)}")

    for i, vol_date in enumerate(list_of_dates):
        high = 0
        low = 0

        # Get the start point from which to measure performance
        start_date = vol_df[(vol_df.unique == vol_date) & (vol_df.volatility <= vol)].iloc[0].time

        #     vol_df[vol_df.volatility<0.06].iloc[0].name+timedelta(days=10)

        end_date = start_date + timedelta(days=int(timeperiod))
        #     print(start_date.date(), end_date.date())
        # Get the check period based on user input
        #     check_period = btc.iloc[start_date.date():end_date.date()]
        check_period = df[df.time.between(start_date, end_date)]
        start_price = round(check_period.iloc[0].close, 3)

        # Now iterate over the specified time period and check the max percentage gain and fall
        for ind, row in check_period.iterrows():

            new_low = row.low
            new_high = row.high
            pct_low = round(((new_low / start_price) - 1) * 100, 3)
            pct_high = round(((new_high / start_price) - 1) * 100, 3)
            if new_low < start_price and pct_low < low:
                low = pct_low
            elif new_high > start_price and pct_high > high:
                high = pct_high

        #         print(f"Threshold breached on {start_date}, at ${start_price}, after {timeperiod} days:")
        #         print(f"Best Gain: {high}%, Best Fall: {low} \n")

        data_point = pd.DataFrame({"thresh_date": start_date,
                                   "price": start_price,
                                   "gain": high,
                                   "fall": low,
                                   "volatility": vol,
                                   "period": timeperiod}, index=[i])
        vol_data = vol_data.append(data_point)

    # Get a list of df indices to be added to the plot
    plot_ind = []
    # get the last accepted plot data
    last_plot_date = 0
    for ind, row in vol_data.iterrows():
        if ind == 0:
            plot_ind.append(ind)
            last_plot_date = row.thresh_date

        # If the current row is within a month of our last plot date - ignore it
        elif row.thresh_date < last_plot_date + timedelta(days=30):
            pass
        # Else keep the plot date and add its index to the plot list
        else:
            plot_ind.append(ind)
            last_plot_date = row.thresh_date

    plot_vol = vol_data.loc[plot_ind]

    # Add a feature to state whether the best trade was a buy or a sell - based on magnitude of price change
    plot_vol["direction"] = plot_vol.apply(lambda x: "buy" if x.gain > abs(x.fall) else "sell", axis=1)

    # Create the on hover information
    # Add hover text
    hover_text = []

    for ind, row in plot_vol.iterrows():
        #     print(row["volatility"])
        hover_text.append(f"Date:{row.thresh_date}<br>" +
                          f"Volatility: {round(row['volatility'], 3)}<br>" +
                          f"Start Price: ${round(row.price)}<br>" +
                          f"%age {row.period} day gain: {row.gain}<br>" +
                          f"%age {row.period} day fall: {row.fall}"

                          )
    plot_vol["text"] = hover_text

    return plot_vol


# Build a confirmation indicator
def build_indicator(df, indicator):

    """
    In this function we'll construct features that respresent a variety of confirmation indicators including:
    1. SMA (200,100,50)
    2. EMA (?)
    3. MACD
    4. STOCHRSI
    """

    # If indicator matches one of our expected then build that feature
    if indicator=="SMA":

        df["sma200"] = ta.sma(close=df["close"], length=200)
        df["sma100"] = ta.sma(close=df["close"], length=100)
        df["sma50"] = ta.sma(close=df["close"], length=50)




    elif indicator=="EMA":
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)
        df["ema200"] = ta.ema(df["close"], length=200)

    elif indicator=="STOCHRSI":
        stoch_rsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        stoch_rsi.rename(columns={"STOCHRSIk_14_14_3_3": "k", "STOCHRSId_14_14_3_3": "d"}, inplace=True)

        # stoch_rsi.reset_index(inplace=True)
        df = df.join(stoch_rsi)
        # df.reset_index(inplace=True)

    elif indicator=="MACD":
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        macd.rename(columns={"MACD_12_26_9": "macd", "MACDh_12_26_9": "histogram", "MACDs_12_26_9": "slow"}, inplace=True)
        # macd.reset_index(inplace=True)

        df = df.join(macd)



    return df



# Build the plotter
def plot_data(df, plot_vol, vol, timeperiod, indicator):
    # SET THE OVERALL FIGURE
    # fig = make_subplots(rows=2, cols=1, subplot_titles=("Signals", "Confirmations"))
    fig = make_subplots(rows=2, cols=1)
    fig.update_layout(
        width=1250,
        height=800,
        yaxis_domain=[0.35, 0.99],
        yaxis2_domain=[0.0, 0.3],  # Changes the aspect ratio of the second plot

    )

    # Hard code the halving dates as we want to display these on the visuals
    halvings = ['2012-11-28', '2016-07-09', '2020-05-11']

    # Get the user generated sell and buy signal data
    sells_df = plot_vol[plot_vol.direction == "sell"]
    buys_df = plot_vol[plot_vol.direction == "buy"]

    # Build the first figure
    # BTC USD
    fig1 = go.Figure(
        go.Line(x=df.time, y=df.close, line_width=2, name='$BTC', text=df["text"], hoverinfo='text')
    )

    # BTC LOWER BB
    fig1.add_trace(
        go.Line(x=df.time, y=df.lower, line_width=1, name="Lower BB Bound", text=df["text"], hoverinfo='text')
    )

    # BTC UPPER BB
    fig1.add_trace(
        go.Line(x=df.time, y=df.upper, line_width=1, name="Upper BB Bound", text=df["text"], hoverinfo='text')
    )

    # Add volatility trace to the main chart
    # Use add_trace function and specify secondary_y axes = True.
    fig1.add_trace(
        go.Line(x=df.time, y=df.volatility, name="Volatility", yaxis="y2", opacity=0.4)
    )

    # Add sell signals
    fig1.add_trace(
        go.Scatter(mode="markers", x=sells_df.thresh_date, y=sells_df.price * 1.1, name="Sell Signal",
                   marker=dict(
                       symbol="arrow-down",
                       color="red",
                       size=10
                   ), text=sells_df.text, hoverinfo="text"))

    # Add buy signals
    fig1.add_trace(
        go.Scatter(mode="markers", x=buys_df.thresh_date, y=buys_df.price * 0.9, name="Buy Signal",
                   marker=dict(
                       symbol="arrow-up",
                       color="MidnightBlue",
                       size=10
                   ), text=buys_df.text, hoverinfo="text"))

    # Add halvings
    for i, line in enumerate(halvings):
        if i == 0:
            fig1.add_trace(
                go.Line(x=[pd.to_datetime(line), pd.to_datetime(line)], y=[0.02, 95000],
                        mode="lines", line={'dash': "dot", 'color': 'FireBrick', 'width': 1}, name="Halving")

            )
        else:
            fig1.add_trace(
                go.Line(x=[pd.to_datetime(line), pd.to_datetime(line)], y=[0.02, 95000],
                        mode="lines", line={'dash': "dot", 'color': 'FireBrick', 'width': 1}, showlegend=False)

            )

    # Add traces to figure
    fig.add_trace(fig1["data"][0], row=1, col=1)
    fig.add_trace(fig1["data"][1], row=1, col=1)
    fig.add_trace(fig1["data"][2], row=1, col=1)
    fig.add_trace(fig1["data"][3], row=1, col=1)
    fig.add_trace(fig1["data"][4], row=1, col=1)
    fig.add_trace(fig1["data"][5], row=1, col=1)
    fig.add_trace(fig1["data"][6], row=1, col=1)
    fig.add_trace(fig1["data"][7], row=1, col=1)
    fig.add_trace(fig1["data"][8], row=1, col=1)

    # PLOT THE SECONDARY INDICATOR
    # fig2 = go.Figure()

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    if indicator=="SMA":
        fig2.add_trace(
            go.Line(x=df.time, y=df.sma200, name="SMA_200")

        )
        fig2.add_trace(
            go.Line(x=df.time, y=df.sma100, name="SMA_100")

        )
        fig2.add_trace(
            go.Line(x=df.time, y=df.sma50, name="SMA_50")

        )



        # Add sell signals
        fig2.add_trace(
            go.Scatter(mode="markers", x=sells_df.thresh_date, y=sells_df.price*1.1, name="Sell Signal",
                       marker=dict(
                           symbol="arrow-down",
                           color="red",
                           size=10
                       ), showlegend=False))

        # Add buy signals
        fig2.add_trace(
            go.Scatter(mode="markers", x=buys_df.thresh_date, y=buys_df.price*0.9, name="Buy Signal",
                       marker=dict(
                           symbol="arrow-up",
                           color="MidnightBlue",
                           size=10
                       ), showlegend=False))


        fig.add_trace(fig2["data"][0], row=2, col=1)
        fig.add_trace(fig2["data"][1], row=2, col=1)
        fig.add_trace(fig2["data"][2], row=2, col=1)
        fig.add_trace(fig2["data"][3], row=2, col=1)
        fig.add_trace(fig2["data"][4], row=2, col=1)


        fig.update_yaxes(type="log")



    elif indicator=="EMA":

        fig2.add_trace(
            go.Line(x=df.time, y=df.ema200, name="EMA_200")

        )
        fig2.add_trace(
            go.Line(x=df.time, y=df.ema50, name="EMA_50")

        )
        fig2.add_trace(
            go.Line(x=df.time, y=df.ema21, name="EMA_21")

        )

        # Add sell signals
        fig2.add_trace(
            go.Scatter(mode="markers", x=sells_df.thresh_date, y=sells_df.price * 1.1, name="Sell Signal",
                       marker=dict(
                           symbol="arrow-down",
                           color="red",
                           size=10
                       ), showlegend=False))

        # Add buy signals
        fig2.add_trace(
            go.Scatter(mode="markers", x=buys_df.thresh_date, y=buys_df.price * 0.9, name="Buy Signal",
                       marker=dict(
                           symbol="arrow-up",
                           color="MidnightBlue",
                           size=10
                       ), showlegend=False))

        fig.add_trace(fig2["data"][0], row=2, col=1)
        fig.add_trace(fig2["data"][1], row=2, col=1)
        fig.add_trace(fig2["data"][2], row=2, col=1)
        fig.add_trace(fig2["data"][3], row=2, col=1)
        fig.add_trace(fig2["data"][4], row=2, col=1)

        fig.update_yaxes(type="log")

    elif indicator=="STOCHRSI":

        fig2.add_trace(
            go.Line(x=df.time, y=df.k, name="STOCHRSI_K")

        )
        fig2.add_trace(
            go.Line(x=df.time, y=df.d, name="STOCHRSI_D")

        )
        # Add sell signals
        sells_data = list(100 for i in range(0, sells_df.shape[0]))
        buys_data = list(0 for i in range(0, buys_df.shape[0]))
        fig2.add_trace(
            go.Scatter(mode="markers", x=sells_df.thresh_date, y=sells_data, name="Sell Signal",
                       marker=dict(
                           symbol="arrow-down",
                           color="red",
                           size=10
                       ), showlegend=False))

        # Add buy signals
        fig2.add_trace(
            go.Scatter(mode="markers", x=buys_df.thresh_date, y=buys_data, name="Buy Signal",
                       marker=dict(
                           symbol="arrow-up",
                           color="MidnightBlue",
                           size=10
                       ), showlegend=False))

        fig.add_trace(fig2["data"][0], row=2, col=1)
        fig.add_trace(fig2["data"][1], row=2, col=1)
        fig.add_trace(fig2["data"][2], row=2, col=1)
        fig.add_trace(fig2["data"][3], row=2, col=1)
        fig.update_yaxes(type="log", row=1, col=1)

    elif indicator=="MACD":

        fig2.add_trace(
            go.Line(x=df.time, y=df.macd, name="MACD")

        )
        fig2.add_trace(
            go.Bar(x=df.time, y=df.histogram, name="Histogram")

        )
        fig2.add_trace(
            go.Line(x=df.time, y=df.slow, name="MACD Oscillator")

        )

        # Add sell signals
        sells_data = list(0 for i in range(0, sells_df.shape[0]))
        buys_data = list(0 for i in range(0, buys_df.shape[0]))

        fig2.add_trace(
            go.Scatter(mode="markers", x=sells_df.thresh_date, y=sells_data, name="Sell Signal",
                       marker=dict(
                           symbol="arrow-down",
                           color="red",
                           size=10
                       ), showlegend=False))

        # Add buy signals
        fig2.add_trace(
            go.Scatter(mode="markers", x=buys_df.thresh_date, y=buys_data, name="Buy Signal",
                       marker=dict(
                           symbol="arrow-up",
                           color="MidnightBlue",
                           size=10
                       ), showlegend=False))

        fig.add_trace(fig2["data"][0], row=2, col=1)
        fig.add_trace(fig2["data"][1], row=2, col=1)
        fig.add_trace(fig2["data"][2], row=2, col=1)
        fig.add_trace(fig2["data"][3], row=2, col=1)
        fig.add_trace(fig2["data"][4], row=2, col=1)
        fig.update_yaxes(type="log", row=1, col=1)






    fig.update_layout(
        xaxis2_title="TIME",
        yaxis_title="BTCUSD 1D CLOSE",
        yaxis2_title=f"{indicator}",
        title=f"Volatility Back Test: volatility < {round(vol,3)} over {timeperiod} days",
        # title= f"Requested Metrics: volatility < {round(vol,3)} over {timeperiod} days",
        title_font_color="#003865"

    )



    # Set the y axis to log type so we can more easily see the historic values
    # fig.update_yaxes(type="log")
    # fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(showgrid=False)
    # fig.update_yaxes(autorange=True, row=2, col=1)
    # Add traces to figure

    # fig.show()

    return fig

