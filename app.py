import pandas as pd
from flask import Flask, request, jsonify
import pickle
import streamlit as st

#load model
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])

def predict():
    interest_rate = st.sidebar.slider('int_rate', 1, 50)
    loan_amount = st.sidebar.slider('loan_amnt', 10000, 40000)
    fico_score = st.sidebar.slider('fico_score', 400, 900)
    term = st.sidebar.selectbox('term', (' 36 months', ' 60 months'))
    data = {'int_rate': interest_rate,
            'loan_amnt': loan_amount,
            'fico_score': fico_score,
            'tem': term}
    df = pd.DataFrame(data, columns=['int_rate', 'loan_amnt', 'fico_score', 'term'], index=[0])
    with open ('model', 'rb') as file:
        model = pickle.load(file)

    result = model.predict(df)
    res = {'result': result}
    return jsonify(res)




if __name__ == "__main__":
 app.run(debug=True)
