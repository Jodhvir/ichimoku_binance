# Use this script to generate Ichimoku cloud chart

import pandas as pd
import json, requests, talib
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import plotly.graph_objects as go

# streamlit run 2_Ichimoku_Crypto_Streamlit.py

def get_data_from_binance(symbol,interval,future):
    if future:
        url = f'''https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}'''
    else:
        url = f'''https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}'''
    
    # url = f'''https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000'''
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['Open Time',
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close Time', 'qav', 'num_trades',
                    'taker_base_vol', 'taker_quote_vol', 'ignore']

    # Convert UNIX time to Normal time
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    # df.set_index('Open Time')

    # Change the type of Close and Volume values to Float to avoid any error
    df['Close'] = pd.to_numeric(df['Close'], downcast="float")
    df['Volume'] = pd.to_numeric(df['Volume'], downcast="float")
    df['Open'] = pd.to_numeric(df['Open'], downcast="float")
    df['Low'] = pd.to_numeric(df['Low'], downcast="float")
    df['High'] = pd.to_numeric(df['High'], downcast="float")

    # Drop all columns except Open Time, Open, Close, High, Low, Volume 
    col_delete = ['Close Time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore']
    df.drop(col_delete, axis = 1, inplace=True)
    # df.drop(df.tail(1).index, inplace=True) # drop last row since the candle has not Closed yet
    return df

'''
Extend the Ichimoku Datarame so that it incudes future values of Leading Span A & Leading Span B
'''
def extend_numeric_index(df, periods=26):
    # Get the last index value
    last_index = df.index[-1]
    
    # Create new index values
    new_indices = range(last_index + 1, last_index + 1 + periods)
    
    # Create a new DataFrame with the new indices
    future_df = pd.DataFrame(index=new_indices, columns=df.columns)
    
    # Concatenate the original DataFrame with the new DataFrame
    extended_df = pd.concat([df, future_df])
    
    return extended_df

def get_ichimoku_cloud(df):
    df = df.copy()

    # Calculate Tenkan-sen (9-period)
    df['Conversion Line'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2

    # Calculate Kijun-sen (26-period)
    df['Base Line'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    df['Leading Span A'] = ((df['Conversion Line'] + df['Base Line']) / 2).shift(26)

    # Calculate Senkou Span B (52-period)
    df['Leading Span B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    
    # Calculate Chikou Span (Lagging Span)
    df['Lagging Span'] = df['Close'].shift(-26)

    # Make Cloud
    conditions = [df['Leading Span A'] > df['Leading Span B']]
    choices = ['Green']
    df['Cloud'] = np.select(conditions, choices, 'Red')
    
    return df

def main():
    # Set page layout to wide
    st.set_page_config(page_title='Ichimoku-Crypto', layout="wide")
    options = [True,False]

    # st.title("Ichimoku Cloud Chart")
    with st.sidebar:
        with st.form(key='Form_1'):
            symbol = str(st.text_input('Enter the Coin-Pair *')).upper()
            interval = st.selectbox('Select interval *', ('1M','1w','3d','1d','12h','8h','6h','4h','2h','1h','30m','15m','5m','3m','1m'))
            n = st.number_input("Enter a number", min_value=0, max_value=500, value=120, step=1)
            selected_option = st.selectbox('Future?', options)
            submit = st.form_submit_button(label='Submit')
    
    if submit:
        # Get data for BTCUSDT first, so that Coorelation can be found
        df_btc = get_data_from_binance('BTCUSDT',interval,selected_option)
        df_btc['Diff_btc'] = df_btc['Close'].pct_change()
        df_btc.dropna(inplace=True)

        # Get data for selected coin
        df = get_data_from_binance(symbol,interval,selected_option)
        df_symbol = df.copy()
        df_symbol['Diff'] = df_symbol['Close'].pct_change()
        df_symbol.dropna(inplace=True)

        merged_df = pd.merge(df_btc, df_symbol, on ='Open Time',how ='inner')
        # print(merged_df.tail())
        coorelation = merged_df['Diff'].corr(merged_df['Diff_btc'])


        # Print Title, Coorelation and current Price

        st.title(f'{symbol} - {interval}: {round(coorelation,1)}')
        current_price = df['Close'].iloc[-1]
        st.subheader(f'''Current Price: {current_price}''')

        # Get data for selected symbol
        
        df['EMA-200'] = talib.EMA(df['Close'],200)
        df['RSI'] = round(talib.RSI(df['Close'], timeperiod=14),1)
        last_rsi = df['RSI'].iloc[-1]
        df = extend_numeric_index(df, periods=26)
        df = get_ichimoku_cloud(df)

        # Get Candles for Last N values only
        df = df.tail(n).copy()

        df.drop('Open Time', axis=1, inplace=True)

        # Create a candlestick chart
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Candlesticks"
        ))

        # EMA-200
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA-200'],
            line=dict(color='#4F6D7A', width=2, dash='dash'), # # solid, dash, dot, dashdot
            name="EMA-200"
        ))

        # Leading Span A
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Leading Span A'],
            line=dict(color='green', width=1),
            name="Leading Span A"
        ))

        # Leading Span B
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Leading Span B'],
            line=dict(color='red', width=1),
            name="Leading Span B"
        ))

        # Make a 3rd Line containing values from cloud
        df['senkou_c'] = np.nan
        for index, row in df.iterrows():
            if row['Leading Span A'] < row['Leading Span B']:
                df.at[index,'senkou_c'] = row['Leading Span A']
            elif row['Leading Span A'] >= row['Leading Span B']:
                df.at[index,'senkou_c'] = row['Leading Span B']
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['senkou_c'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Leading Span B'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            fillcolor='rgba(255, 0, 0, 0.2)',  # Green Cloud
            name="Ichimoku Cloud (Green)"
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['senkou_c'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Leading Span A'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            fillcolor='rgba(0, 255, 0, 0.2)',  # Red Cloud
            name="Ichimoku Cloud (Green)"
        ))

        # Conversion Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Conversion Line'],
            line=dict(color='blue', width=2),
            name="Conversion Line"
        ))

        # Base Line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Base Line'],
            line=dict(color='purple', width=2),
            name="Base Line"
        ))

        # Lagging Span
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Lagging Span'],
            line=dict(color='black', width=2, dash='dot'), # solid, dash, dot, dashdot
            name="Lagging Span"
        ))

        # Customize layout
        fig.update_layout(
            # title=interval,
            yaxis_title="Price",
            xaxis_title="Time",
            xaxis_rangeslider_visible=False,
            showlegend=False
        )

        # Show the plot in Streamlit
        st.plotly_chart(fig)

        
        # Make RSI Chart
        RSI_LINE_COLOR = '#EF0AFF'
        # RSI_LINE_COLOR = '#F573FF'
        # st.subheader(f'''RSI - {last_rsi}''')
        fig2 = go.Figure()

        # Shade the area between 25 and 85
        fig2.add_shape(
            type="rect",
            x0=df.index[0], x1=df.index[-1],  # Full width of the chart
            y0=25, y1=85,  # Y-axis range for shading
            fillcolor=RSI_LINE_COLOR, opacity=0.1, line_width=0)
        
        fig2.add_trace(go.Scatter(
            x=df.index, 
            y=df['RSI'], 
            line=dict(color=RSI_LINE_COLOR, width=2), # solid, dash, dot, dashdot, 
            name='RSI'
            ))
        

        # fig2.update_layout(height=300)
        fig2.update_layout(title=f'''RSI - {last_rsi}''', height=300)
        # fig2.update_layout(title='RSI',width=600, height=300)
        st.plotly_chart(fig2)

        st.dataframe(df)

if __name__ == '__main__':
    main()