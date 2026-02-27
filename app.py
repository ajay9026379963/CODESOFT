import streamlit as st 
import pandas as pd
import joblib

# load the trained model 
model = joblib.load('linear_model.pkl')

# user input frame 

st.title('IIRS flower prediction app')

sepal_lenth = st.slider('sepal length')
sepal_weight = st.slider('sepal weight')
petal_length = st.slider('petal length')
petal_weight = st.slider('petal weight')

# convert categorical input to numerical using label encoder
encoder = joblib.load('label_encoder.pkl')



# prepare input row for prediction 

data = pd.DataFrame({
   'sepal_length': [sepal_lenth],
   'sepal_width': [sepal_weight],
   'petal_length': [petal_length],
   'petal_width': [petal_weight] 
})

# make prediction

if st.button('Predict'):
    prediction = model.predict(data)[0]
    st.write(f'The prediction flower is {prediction}')
    st.success('Prediction successfully completed')
    
