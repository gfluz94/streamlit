import streamlit as st
import pandas as pd
import numpy as np


if __name__ == "__main__":

    st.write("""
    # Simple Stock Price App

    Below we can visualize the stock closing price and volume of Google!
    
    """)

    data = np.random.normal(loc=152.0, scale=38.5, size=(1000, 2))
    data_range = pd.date_range(end="2021-09-06", periods=1000)
    ticker_df = pd.DataFrame(data, columns=["Close", "Volume"], index=data_range)

    st.line_chart(ticker_df.iloc[:, 0])
    st.line_chart(ticker_df.iloc[:, 1])