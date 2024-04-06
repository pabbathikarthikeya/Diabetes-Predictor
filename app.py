import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__, static_folder='templates')

sc = pickle.load(open('sc.pkl', 'rb'))
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        pred = model.predict(sc.transform(final_features))
        return render_template('result.html', prediction=pred)
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)
