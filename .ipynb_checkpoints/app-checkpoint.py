import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \
        (sentosa, versicolor, virginica) based on their sepal/petal \
        and lenght/width.')

st,header("Plant Features")
col1, col2 = st.column(2)

with col1:
    st.text("Sepal characteristics")
    sepal_1 = st.slider('Sepal length (cm)', 1.0, 7.0, 0.5)
    sepal_w = st.slider('Sepal width (cm) ', 2.0, 2.5, 0.5)

with col2:
    st.test("Petal characteristics")
    petal_1 = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 2.0, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    result = predict(
        np.array([[sepal_1, sepal_w, petal_1, petal_w]]))
    st.text(result[0])

st.text('')
st.text('')
st.markdown(
    "This model was created for eductaional purpose only."
)