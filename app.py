from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime
import pandas as pd

pipeline = joblib.load('weather_model.pkl')

app = Flask(__name__)

# Define the prediction function
def predict_weather(City, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    features = pd.DataFrame({
        'City': [City],
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'day_of_week': [date.weekday()]
    })
    prediction = pipeline.predict(features)
    precip_mm = prediction[0][3]
    chance_of_rain = min(max(precip_mm / 10, 0), 100)
    return {
        'tavg': prediction[0][0],
        'tmin': prediction[0][1],
        'tmax': prediction[0][2],
        'chance_of_rain_percentage': chance_of_rain
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/current.html')
def current_weather():
    return render_template('current.html')

@app.route('/aboutus.html')
def about_us():
    return render_template('aboutus.html')

@app.route('/logout.html')
def logout():
    # Handle logout logic here
    return render_template('logout.html')


@app.route('/predict', methods=['GET', 'POST'])  # Allow GET and POST
def predict():
    if request.method == 'POST':
        city = request.form['city']
        date = request.form['date']
        # Call your ML model's predict_weather function
        result = predict_weather(city, date)
        # Extract values and render the template
        tavg = result['tavg']
        tmin = result['tmin']
        tmax = result['tmax']
        precipitation = result['chance_of_rain_percentage']
        return render_template('predict.html', city=city, date=date, tavg=tavg, tmin=tmin, tmax=tmax, precipitation=precipitation)
    else:
        # Default values for GET requests
        return render_template('predict.html', city=None, date=None, tavg=None, tmin=None, tmax=None, precipitation=None)



if __name__ == '__main__':
    app.run(debug=True)
