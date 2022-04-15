import pandas as pd
from flask import Flask, render_template
import pickle
import streamlit as st

#load model
app = Flask(__name__)



@app.route('/', methods=['POST'])

def user_input_features():

 interest_rate = st.sidebar.slider('int_rate', 1, 50)
 loan_amount = st.sidebar.slider('loan_amnt', 10000, 40000)
 fico_score = st.sidebar.slider('fico_score', 400, 900)
 term = st.sidebar.selectbox('term', (' 36 months', ' 60 months'))
 data = {'int_rate': interest_rate,
         'loan_amnt': loan_amount,
         'fico_score': fico_score,
         'tem': term}
 features = pd.DataFrame(data, columns = ['int_rate', 'loan_amnt', 'fico_score', 'term'], index=[0])
 return features

@app.route('/')
def predict(df):
 result = model.predict(df)

 return result

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
model = pickle.load(open('model.pkl','rb'))
prediction = model.predict(df)
#prediction_proba = model.predict_proba(df)

st.subheader('Prediction')
st.write(prediction)

#st.subheader('Default Probability')
#st.write(prediction_proba)


if __name__ == "__app__":
 app.run(debug=True)