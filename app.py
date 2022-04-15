import pandas as pd
from flask import Flask, jsonify, request
import pickle
import streamlit as st

#load model
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)


 # routes
@app.route('/predict')
def predict(features):
 result = model.predict(features)

 return result


@app.route('/user_input_features', methods=['POST'])

def user_input_features():

 interest_rate = st.sidebar.slider('int_rate', 1, 50)
 loan_amount = st.sidebar.slider('loan_amnt', 10000, 40000)
 fico_score = st.sidebar.slider('fico_score', 400, 900)
 term = st.sidebar.selectbox('term', (' 36 months', ' 60 months'))
 data = {'int_rate': interest_rate,
         'loan_amnt': loan_amount,
         'fico_score': fico_score,
         'tem': term}
 features = pd.DataFrame(data, index=None)
 return features
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
model = pickle.load(open('model.pkl','rb'))
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)






if __name__ == "__main__":
 app.run()
