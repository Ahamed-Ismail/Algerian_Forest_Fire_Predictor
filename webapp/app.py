from flask import Flask,request,jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

appplication=Flask(__name__)
app=appplication

#import ridge regressor and standardscaler model from pickle file

ridge_regressor=pickle.load(open('models/ridgereg.plk','rb'))
standard_scaler=pickle.load(open('models/scalermodel.plk','rb'))




@app.route("/",methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_regressor.predict(new_data_scaled)
        return render_template('index.html', result=round(result[0],4))

    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=3000, debug=True)