import io
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime


@st.cache
def load_data(year: int) -> pd.DataFrame:
    df = pd.read_html(f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html")[0]
    df = (
        df.loc[df.Age != "Age", :]
          .fillna(0)
          .drop(["Rk"], axis=1)
    )
    return df

def download_file(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

def plot_heatmap(df: pd.DataFrame):
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("dark"):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, mask=mask, vmax=1, square=True, ax=ax)
    return fig


st.title("NBA Players Stats Explorer")
st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Data Source:** [Basketball-Reference](https://www.basketball-reference.com/)
""")

st.sidebar.header("Input Parameters")
selected_year = st.sidebar.selectbox("Year", range(datetime.now().year, 1949, -1))
df = load_data(selected_year)

available_teams = df.Tm.unique()
available_positions = df.Pos.unique()

selected_teams = st.sidebar.multiselect("Team", available_teams, default=available_teams)
selected_positions = st.sidebar.multiselect("Position", available_positions, default=available_positions)

df_display = df.loc[(df.Tm.isin(selected_teams)) &
                    (df.Pos.isin(selected_positions)), :]

st.header("Display Player Stats")
st.write(f"Table's dimension: {len(df)} rows and {df.shape[1]} columns")
st.dataframe(df_display)
st.write("***")
st.markdown(download_file(df_display), unsafe_allow_html=True)

if st.button("Exhibit HeatMap"):
    st.header("HeatMap")
    entry = io.StringIO()
    entry.write(df_display.to_csv(index=False))
    entry.seek(0)
    df_corrected_types = pd.read_csv(entry)
    fig = plot_heatmap(df_corrected_types)
    st.pyplot(fig)