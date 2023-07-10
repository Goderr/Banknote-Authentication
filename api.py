
from flask import Flask,request, render_template, url_for,redirect
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open('model.pkl','rb')
model = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict_note():
    if request.method == "POST":
        variance =float(request.form['Variance'])
        skewness = float(request.form['Skewness'])
        curtosis = float(request.form['curtosis'])
        entropy = float(request.form['entropy'])
        prediction = model.predict([[variance,skewness,curtosis,entropy]])
        pred = prediction[0]
    return redirect(url_for('result', prediction=pred))

@app.route('/result/<int:prediction>')
def result(prediction):
    res = " "
    if prediction == 0:
        res = "not authenticate"
    else:
        res = "authenticate"
    return render_template('result.html', result=res)

@app.route('/predict_file', methods = ["POST"])
def predict_file():
    df = pd.read_csv(request.files.get("file"))
    prediction = model.predict(df)
    return "predicted values are " + str(prediction)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port=5000)
