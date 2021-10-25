import streamlit as st
import altair as alt
from utils import barplot_seaborn, read_image, DNA

st.image(read_image("dna_image.jpeg"), use_column_width=True)
st.write("""
## DNA Summarizer

This application counts the nucleotide composition of a query DNA and also the mRNA transcription.
""")

st.header("Enter DNA sequence below:")
sequence_input = "> DNA Query:\nGAACACGTGCAG"
sequence = st.text_area("Sequence Input", sequence_input, height=20)
dna = DNA(sequence)

st.write("***")

st.header("Results")

st.subheader("1. Count Dictionary")
st.write(dna.count_nucleotides())

st.subheader("2. Count Text")
for text in dna.explanatory_text():
    st.write(text)

st.subheader("3. Table")
st.write(dna.get_dataframe())

st.subheader("4. Bar Plot")
st.bar_chart(dna.get_dataframe())

st.subheader("5. Bar Plot - Seaborn")
st.pyplot(barplot_seaborn(dna))

st.subheader("6. Associated m-RNA Sequence")
st.write(dna.get_mrna())