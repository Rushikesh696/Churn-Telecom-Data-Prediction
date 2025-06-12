import streamlit as st
import numpy as np
import pickle
import sklearn
st.title("Iris Species Prediction")
sl=st.text_input("Enter the sepal length (cm):")
sw=st.text_input("Enter the sepal width (cm):")
pl=st.text_input("Enter the petal length (cm):")
pw=st.text_input("Enter the petal width (cm):")

btn_click=st.button("Predict")



with open("Iris_model.pkl", "rb") as f:
    model=pickle.load(f)

if btn_click == True:
    arr=np.array([[float(sl), float(sw), float(pl), float(pw)]]).reshape(1,-1)
    output= model.predict(arr)
    st.write(output)
