import streamlit as st
import pandas as pd 
import joblib
import numpy as np

# Load the pre trained model
model= joblib.load('credit_card_froud.pkl')
st.title("Credit Card Fraud Detection")

# Get user input features 
input_df = st.text_input("Enter all required features:")
input_df_splited = input_df.split(",")

submit = st.button('Submit')

if submit:
    feature = np.array(input_df_splited, dtype= np.float64)
    prediction = model.predict(feature.reshape(1, -1)) 
    
    if prediction[0] == 0:
        st.write("The transaction is Not Fraudulent")
    else:
        st.write("The transaction is Fraudulent")