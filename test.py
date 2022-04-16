import streamlit as st
import pandas as pd
import pickle
st.write("""
# Loan Default Prediction App
This app predicts if a loan defaults or not!
""")

st.sidebar.header('User Input Features')
def input_data():
    ir = st.sidebar.slider('int_rate', 3, 100, 15)
    la = st.sidebar.slider('loan_amount', 1000, 40000, 5000)
    fs = st.sidebar.slider('fico_score', 150, 900, 650)
    term = st.sidebar.selectbox('term', (' 36 months', ' 60 months'))
    data = {'int_rate': ir,
            'loan_amnt': la,
            'fico_score': fs,
            'term': term}
    df = pd.DataFrame(data, columns=['int_rate', 'loan_amnt', 'fico_score', 'term'], index=[0])
    return df

def predict_point(df):
    model = pickle.load(open('model.pkl', 'rb'))
    result = model.predict(df)
    probability = model.predict_proba(df)
    return result, probability
df = input_data()
response, prob = predict_point(df)
st.subheader('Prediction')
st.write(response)

st.subheader('Probability of default')
st.write(prob[0][0])

