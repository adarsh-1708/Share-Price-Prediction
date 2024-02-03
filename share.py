from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('Prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("Frontend.html", prediction_text="")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        industry = request.form['industry']
        t1 = float(request.form['t1'])
        t2 = float(request.form['t2'])
        t3 = float(request.form['t3'])

        # Assuming you have a function to process the input features
        final_features = np.array([[t1, t2, t3]])

        # Assuming you have a function to get the prediction
        prediction = model.predict_proba(final_features)[0][1]

        return render_template('Frontend.html', prediction_text=f"Predicted probability of fire occurrence: {prediction:.2%}")

    return render_template('Frontend.html', prediction_text="")

if __name__ == '__main__':
    app.run(debug=True)
