from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split

from display_utils import binary_classification_evaluation_plot, evaluate_classification


TEST_SIZE = 0.25
SEED = 99


@st.cache
def load_data() -> Tuple[pd.DataFrame, dict]:
    data = load_breast_cancer(as_frame=True)
    idx2target = {idx:name for idx, name in enumerate(data["target_names"])}
    df = data["frame"]
    return df, idx2target

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame]:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=TEST_SIZE,
                                                        random_state=SEED)
    return (X_train, X_test, y_train, y_test)

def display_tree(model, feature_names, class_names):
    fig, _ = plt.subplots(1, 1, figsize=(20, 10))
    plot_tree(model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            fontsize=14)
    return fig

def display_feature_importance(model, feature_names):
    df = pd.DataFrame({
        "Importance": model.feature_importances_,
        "Feature": feature_names
    })
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sns.barplot(x="Importance", y="Feature", data=df.sort_values(by="Importance", ascending=False))
    return fig

st.title("Classification Task Report")
st.markdown("""
This app is a user-friendly interface to further evaluate a tree-based model and understand its rules.
Moreover, it is possible to change hyperparameters and study its predictions.
""")

df, idx2target = load_data()
X_train, X_test, y_train, y_test = split_data(df)

st.sidebar.header("Input Parameters")
st.sidebar.subheader("Model Hyperaparameters")
max_depth = st.sidebar.slider("Tree Maximum Depth", min_value=1, max_value=5, value=3)
min_samples_leaf = st.sidebar.slider("Minimum Samples Leaf", min_value=4, max_value=50, value=10)
threshold = st.sidebar.slider("Classifier Threshold", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
st.sidebar.subheader("Input Example")
allowable_inputs = {k: (float(df.loc[:, k].min()), float(df.loc[:, k].max()))\
                    for k in df.columns[:-1]}
inputs = dict()
for field, (min_value, max_value) in allowable_inputs.items():
    inputs[field] = st.sidebar.slider(field.capitalize(),
                                      min_value=min_value,
                                      max_value=max_value,
                                      value=(min_value+max_value)/2)

dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED)
dt.fit(X_train, y_train)

st.header("Trained Tree Model")
st.subheader("Rules")
st.text(export_text(dt, feature_names=list(X_train.columns)))
st.subheader("Tree Nodes")
fig = display_tree(model=dt,
                   feature_names=list(X_train.columns),
                   class_names=list(idx2target.values()))
st.pyplot(fig)

if st.button("Exhibit Feature Importance"):
    st.subheader("Feature Importance")
    fig = display_feature_importance(dt, feature_names=list(X_train.columns))
    st.pyplot(fig)

st.write("***")

st.header("Test Set Evaluation")
y_proba = dt.predict_proba(X_test)[:, 1]
y_pred = np.int8(y_proba > threshold)
st.subheader("Performance Metrics")
st.dataframe(evaluate_classification(y_test, y_pred, y_proba))

if st.button("Exhibit Evaluation Plots"):
    st.subheader("Performance Plots")
    fig = binary_classification_evaluation_plot(dt, X_test, y_test)
    st.pyplot(fig)

st.write("***")

st.header("Single Instance Prediction")
input_array = list(inputs.values())
example_proba = dt.predict_proba(np.array(input_array).reshape((1, -1)))[0, 1]
class_predicted = idx2target[int(example_proba > threshold)]
st.text(f"Instance's predicted probability of {idx2target[1]} tumor: {100*example_proba:.2f}%")
st.text(f"Predicted class: {class_predicted} (Threshold = {100*threshold:.0f}%)")