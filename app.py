from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import mlflow
import mlflow.sklearn

app = Flask(__name__)
model = pickle.load(open('linear_regression_model.pkl', 'rb'))


form_values = {}
prediction_text = None


mlflow.set_tracking_uri("http://127.0.0.1:5000")  
mlflow.set_experiment("Used Car Price Prediction") 

@app.route('/')
def home():
    return render_template('index.html', form_values=form_values, prediction_text=prediction_text)

@app.route('/predict', methods=['POST'])
def predict():
    global form_values, prediction_text
    
    
    form_values = {key: request.form[key] for key in request.form}

    
    on_road_old = float(request.form.get('on_road_old', 0))
    on_road_now = float(request.form.get('on_road_now', 0))
    economy = float(request.form.get('economy', 0))
    condition = float(request.form.get('condition', 0))
    rating = float(request.form.get('rating', 0))
    

    
    input_features = np.array([[on_road_old, on_road_now, economy, condition, rating]])

    
   


    with mlflow.start_run() as run:
        
        mlflow.log_param("on_road_old", on_road_old)
        mlflow.log_param("on_road_now", on_road_now)
        mlflow.log_param("economy", economy)
        mlflow.log_param("condition", condition)
        mlflow.log_param("rating", rating)
       

        
        prediction = model.predict(input_features)

        
        mlflow.log_metric("prediction", prediction[0])

        
        mlflow.sklearn.log_model(model, "linear_regression_model")

        
        prediction_text = f'₹{prediction[0]:,.2f}'

    return render_template('index.html', form_values=form_values, prediction_text=prediction_text)

@app.route('/clear', methods=['POST'])
def clear():
    global form_values, prediction_text
    
    
    form_values = {}
    prediction_text = None

    return render_template('index.html', form_values=form_values, prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
