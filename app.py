# importing the libareis 
from flask import Flask,request,jsonify,render_template,url_for
import numpy as np

# importing machine learning model
import pickle


# creating the app 
app = Flask(__name__)

# load machine learning model form pickle 
model = pickle.load(open('my_model.pkl','rb'))

 # this is a flask deocrator connect the url
@app.route('/')  # to the python function
def home():
    return render_template('home.html')
    
    
@app.route('/predict',methods = ['POST'])
def predict():
    try:
        input_data = [float(x) for x in request.form.values()]
        final_input = np.array(input_data).reshape(1,-1)
        prediction = model.predict(final_input)[0]
        return render_template('home.html',prediction_text = f"Calories burned = {prediction:.2f} ")
    except Exception as e:
        return render_template('home.html',prediction_text = f"Oops! Error{str(e)}")


@app.route('/predict_api',methods = ['POST'])
def predict_api():
    try:
        data=request.get_json(force = True)['data']
        input_data = np.array(list(data.values())).reshape(1,-1)
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction':float(prediction)})
    except Exception as e:
        return jsonify({'error':str(e)})

if __name__ == '__main__':
    app.run(debug = True,host = '0.0.0.0',port = 8080)
