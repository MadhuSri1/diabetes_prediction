from flask import Flask, render_template, request
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init('YOUR_DSN_HERE', integrations=[FlaskIntegration()])
import pickle
import numpy as np



filename = 'classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/')

def home():
    
    return render_template('ditect.html')  

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        pred = classifier.predict(data)
        
        return render_template('result.html', prediction=pred)


if __name__ == '__main__':
	app.run(debug=True)
