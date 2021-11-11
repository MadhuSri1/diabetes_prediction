import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
filename = 'diabetes-prediction.pkl'
classifier = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('ditect.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    float_features = [float(X) for X in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model.predict( scaler.transform(final_features) )
    return render_template('result.html', prediction = pred)

if __name__ == "__main__":
    app.run(debug=True)
