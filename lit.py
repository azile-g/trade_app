import datetime
import pandas as pd
import numpy as np
import copy

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

import momentum as mm

rsi_tab, macd_tab, bb_tab, data_tab = st.tabs(["📈 RSI", "📈 MACD", "📈 Bollinger Band", "🗃 Raw Data"])

#sidebar
dji_df = mm.fetch_info()
tickers = dji_df.Symbol.values.tolist()
option = st.sidebar.selectbox('Select one symbol', tickers)
today = datetime.date.today()
before = today - datetime.timedelta(days=700)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')
df = yf.download(option, start = start_date, end = end_date, progress=False)

#RSI INDICATOR
rsi_tab.write("# RSI Momentum Trading Strategy:")
rsi = mm.RSIIndicator(df['Close']).rsi()
rsi_df = pd.concat([df, rsi], axis = 1).reset_index()
rsi_fig = px.line(rsi_df, x="Date", y="rsi")
rsi_thres = rsi_tab.slider("Select your RSI upper and lower thresholds:", 0, 100, (30, 70))
rsi_fig.add_hline(rsi_thres[0])
rsi_fig.add_hline(rsi_thres[1])
rsi_tab.subheader(f"{option} RSI Indicator")
rsi_tab.plotly_chart(rsi_fig)
## Calculate the buy & sell signals
rsi_tab.subheader(f"{option} RSI Trading Strategy Results:")
ret_rsi_df = mm.RSIIndicator(df['Close']).trading_signals(rsi_df, rsi_thres[0], rsi_thres[1])
buy_trade_count = ret_rsi_df['Buy Signal'].count()
sell_trade_count = ret_rsi_df['Sell Signal'].count()
#average_profit = ((ret_rsi_df['Strategy'].tolist()[-1] / ret_rsi_df['Strategy'].tolist()[15])**(1/trade_count))-1
total_days = ret_rsi_df['Long Tomorrow'].count()
if buy_trade_count <= 0: 
    average_days = 0
else: 
    average_days = int(total_days / buy_trade_count)
buy_df = ret_rsi_df[["Date", "Buy Signal"]].dropna()
sell_df = ret_rsi_df[["Date", "Sell Signal"]].dropna()
try:
    profits = [(sell_df["Sell Signal"].tolist()[i] - ialue)/ialue for i, ialue in enumerate(buy_df["Buy Signal"].tolist())]
except:
    profits = [(sell_df["Sell Signal"].tolist()[i] - ialue)/ialue for i, ialue in enumerate(buy_df["Buy Signal"].tolist()[:-1])]
#Write results to streamlit
rsi_tab.write(buy_df)
rsi_tab.write(sell_df)
if len(profits) == 0: 
    avrg_profit = 0
else:
    avrg_profit = (sum(profits)/len(profits))*100
rsi_tab.markdown(
    f"""
    - 🤙 Strategy yielded **{buy_trade_count} buy trades** and **{sell_trade_count} sell trades**
    - ⏳ Average trade lasted **{average_days} days per trade**
    - 💵 Our profits per trade was **{profits}**
    - 💵 Our average profit was **{avrg_profit}%**
    """)

#MACD INDICATOR
macd_tab.write("# MACD Momentum Trading Strategy:")
macd = pd.concat([mm.MACD(df['Close']).macd(), mm.MACD(df['Close']).macd_signal()], axis = 1)
macd_df = pd.concat([df, macd], axis = 1).reset_index()
macd_fig = px.area(macd_df, x="Date", y=["MACD_12_26", "MACD_sign_12_26"])
macd_tab.subheader(f"{option} MACD Indicator")
macd_tab.plotly_chart(macd_fig)
trades_df, profits = mm.MACD(df['Close']).trading_signal(macd_df, 1)
macd_tab.subheader(f"{option} MACD Trading Strategy Results:")
macd_tab.write(trades_df)
profits_df = pd.DataFrame(data = {"Date": trades_df["Date"].tolist()[1:], "Profits": profits})
macd_fig = px.line(profits_df, x = "Date", y = "Profits", title = "Rolling Profitability per Trade Placed:")
macd_tab.plotly_chart(macd_fig)

#BOLLINGER BENCHMARK
indicator_bb = mm.BollingerBands(df['Close'])
bb_df = copy.copy(df)
bb_df['High Band'] = indicator_bb.bollinger_hband()
bb_df['Low Band'] = indicator_bb.bollinger_lband()
bb = bb_df[['Close', 'High Band', 'Low Band']].reset_index()
bb_fig = px.line(bb, x = "Date", y = ["Close", "High Band", "Low Band"], title = "Bollinger Bands")
bb_tab.plotly_chart(bb_fig)

#Raw data navigator
data_tab.write('Recent data:')
data_tab.dataframe(df.tail(10))
data_tab.markdown(mm.get_table_download_link(rsi_df, filename = "RSI"), unsafe_allow_html=True)
data_tab.markdown(mm.get_table_download_link(macd_df, filename = "MACD"), unsafe_allow_html=True)
data_tab.markdown(mm.get_table_download_link(bb_df, filename = "Bollinger Band"), unsafe_allow_html=True)


