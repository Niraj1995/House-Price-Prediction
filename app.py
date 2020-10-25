import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("Model.pkl","rb"))
scalar = pickle.load(open("scalar.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ["POST"])
def predict():
    ...
    #For rendering result on HTML GUI
    ...
    
    int_features = [float(x) for x in request.form.values()]

    final_features = [(np.array(int_features))]
    final_features[0][7] = np.log(final_features[0][7])
    final_features[0][8] = np.log(final_features[0][8])
    
    final_features = scalar.transform(final_features)
    
    
    prediction =  model.predict(final_features)
    
    
    output = round(prediction[0], 2)
    output = round(np.expm1(output),2)

    return render_template('index.html',prediction_text = "The House Sale Price is Rs {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
    