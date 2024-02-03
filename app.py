from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('Prediction.pkl', 'rb'))
modeltcs=pickle.load(open('Prediction1.pkl','rb'))
modelinfy=pickle.load(open('PredictionINFY.pkl','rb'))
modeltechm=pickle.load(open('PredictionTECHM.pkl','rb'))

@app.route('/')
def home():
    return render_template("/Frontend.html", prediction_text="")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        industry = request.form['industry']
        if(industry=='Wipro'):
         t1 = float(request.form['t1'])
         t2 = float(request.form['t2'])
         t3 = float(request.form['t3'])

         final_features = np.array([[t1, t2, t3]])
         prediction=model.predict(final_features)
        
         return render_template('/Frontend.html',predicted_price=prediction)
        
        if(industry=='Tata Consulting Services'):
         t1 = float(request.form['t1'])
         t2 = float(request.form['t2'])
         t3 = float(request.form['t3'])

         final_features = np.array([[t1, t2, t3]])
         prediction=modeltcs.predict(final_features)
        
         return render_template('/Frontend.html',predicted_price=prediction)
        
        if(industry=='Tech Mahindra'):
         t1 = float(request.form['t1'])
         t2 = float(request.form['t2'])
         t3 = float(request.form['t3'])

         final_features = np.array([[t1, t2, t3]])
         prediction=modeltechm.predict(final_features)
        
         return render_template('/Frontend.html',predicted_price=prediction)
        
        if(industry=='Infosys'):
         t1 = float(request.form['t1'])
         t2 = float(request.form['t2'])
         t3 = float(request.form['t3'])

         final_features = np.array([[t1, t2, t3]])
         prediction=modelinfy.predict(final_features)
        
         return render_template('/Frontend.html',predicted_price=prediction)
        
if __name__ == '__main__':
    app.run(debug=True)
