# from flask import Flask, render_template, request, redirect
# from markupsafe import Markup
# import numpy as np
# import pandas as pd
# import requests
# import pickle
# import os
# import config

# from utils.fertilizer import fertilizer_dic
# from utils.weather_utils import get_disease_alerts

# # ------------------------- FLASK SETUP -------------------------------------------------
# app = Flask(__name__)
# weather_api_key = config.weather_api_key

# # ------------------------- LOAD MODELS -------------------------------------------------
# crop_recommendation_model = pickle.load(open('models/RandomForest.pkl', 'rb'))

# # ------------------------- UTILITY FUNCTIONS -------------------------------------------------
# def weather_fetch(city_name):
#     """Fetch weather data from OpenWeatherMap API"""
#     base_url = "http://api.openweathermap.org/data/2.5/weather?"
#     complete_url = f"{base_url}appid={weather_api_key}&q={city_name}&units=metric"
    
#     try:
#         response = requests.get(complete_url)
#         response.raise_for_status()
#         data = response.json()
        
#         if "main" in data:
#             return {
#                 'temperature': data['main']['temp'],
#                 'humidity': data['main']['humidity'],
#                 'conditions': data['weather'][0]['main'] if data.get('weather') else 'N/A'
#             }
#         return None
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Weather API Error: {str(e)}")
#         return None

# # ------------------------- ROUTES -------------------------------------------------
# @app.route('/get_weather_data')
# def get_weather_data():
#     city = request.args.get('city')
#     if not city:
#         return jsonify({'error': 'City parameter is required'}), 400
    
#     weather_data = weather_fetch(city)
#     if not weather_data:
#         return jsonify({'error': 'Could not fetch weather data'}), 500
    
#     return jsonify({
#         'temperature': weather_data['temperature'],
#         'humidity': weather_data['humidity']
#     })

# @app.route('/')
# def home():
#     return render_template('index.html', title='Harvestify - Home')

# @app.route('/disease-check', methods=['GET', 'POST'])
# def disease_check():
#     """Handle both weather-based disease alerts and weather result display"""
#     if request.method == 'POST':
#         city = request.form.get('city')
#         if not city:
#             return render_template('disease.html', error="Please enter a city name")
        
#         weather_data = weather_fetch(city)
#         if not weather_data:
#             return render_template('disease.html', error="Could not fetch weather data")
        
#         alerts = get_disease_alerts(
#             weather_data['temperature'], 
#             weather_data['humidity']
#         )
        
#         return render_template(
#             'disease.html',
#             title='Harvestify - Disease Alerts',
#             weather={
#                 'city': city,
#                 'temperature': weather_data['temperature'],
#                 'humidity': weather_data['humidity'],
#                 'conditions': weather_data['conditions']
#             },
#             alerts=alerts
#         )
    
#     return render_template('disease.html', title='Harvestify - Disease Alerts')

# @app.route('/crop-recommend')
# def crop_recommend():
#     return render_template('crop.html', title='Harvestify - Crop Recommendation')

# @app.route('/fertilizer')
# def fertilizer_recommendation():
#     return render_template('fertilizer.html', title='Harvestify - Fertilizer Suggestion')

# @app.route('/crop-predict', methods=['POST'])
# def crop_prediction():
#     """Handle crop prediction based on soil and weather conditions"""
#     try:
#         # Get form data
#         N = int(request.form['nitrogen'])
#         P = int(request.form['phosphorous'])
#         K = int(request.form['pottasium'])
#         ph = float(request.form['ph'])
#         rainfall = float(request.form['rainfall'])
#         city = request.form.get('city')
        
#         # Get weather data
#         weather_data = weather_fetch(city)
#         if not weather_data:
#             return render_template('try_again.html', title='Harvestify - Crop Recommendation')
        
#         # Make prediction
#         data = np.array([[N, P, K, weather_data['temperature'], 
#                          weather_data['humidity'], ph, rainfall]])
#         prediction = crop_recommendation_model.predict(data)[0]
        
#         return render_template(
#             'crop-result.html',
#             prediction=prediction,
#             temperature=weather_data['temperature'],
#             humidity=weather_data['humidity'],
#             city=city,
#             title='Harvestify - Crop Recommendation'
#         )
#     except Exception as e:
#         print(f"Error in crop prediction: {str(e)}")
#         return render_template('try_again.html', title='Harvestify - Crop Recommendation')

# @app.route('/fertilizer-predict', methods=['POST'])
# def fert_recommend():
#     """Provide fertilizer recommendations based on crop needs"""
#     try:
#         crop_name = str(request.form['cropname']).strip().lower()
#         N = int(request.form['nitrogen'])
#         P = int(request.form['phosphorous'])
#         K = int(request.form['pottasium'])
        
#         # Load fertilizer data
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         df = pd.read_csv(os.path.join(base_dir, 'Data', 'fertilizer.csv'))
#         df['Crop'] = df['Crop'].str.strip().str.lower()
        
#         if crop_name not in df['Crop'].values:
#             return render_template('fertilizer-result.html', 
#                                  recommendation="Crop not found in database",
#                                  title='Harvestify - Fertilizer Suggestion')
        
#         # Calculate nutrient differences
#         nr = df[df['Crop'] == crop_name]['N'].iloc[0]
#         pr = df[df['Crop'] == crop_name]['P'].iloc[0]
#         kr = df[df['Crop'] == crop_name]['K'].iloc[0]
        
#         n_diff = nr - N
#         p_diff = pr - P
#         k_diff = kr - K
        
#         # Determine recommendation
#         max_diff = max(abs(n_diff), abs(p_diff), abs(k_diff))
#         if abs(n_diff) == max_diff:
#             key = 'NHigh' if n_diff < 0 else 'Nlow'
#         elif abs(p_diff) == max_diff:
#             key = 'PHigh' if p_diff < 0 else 'Plow'
#         else:
#             key = 'KHigh' if k_diff < 0 else 'Klow'
        
#         recommendation = Markup(str(fertilizer_dic[key]))
#         return render_template('fertilizer-result.html', 
#                              recommendation=recommendation,
#                              title='Harvestify - Fertilizer Suggestion')
#     except Exception as e:
#         print(f"Error in fertilizer recommendation: {str(e)}")
#         return render_template('fertilizer-result.html',
#                              recommendation="Error processing your request",
#                              title='Harvestify - Fertilizer Suggestion')

# # ------------------------- MAIN -------------------------------------------------
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5050, debug=True)

from flask import Flask, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import pickle
import os

from utils.fertilizer import fertilizer_dic
from utils.weather_utils import get_disease_alerts
import config

# ------------------------- FLASK SETUP -------------------------------------------------
app = Flask(__name__)
weather_api_key = config.weather_api_key

# ------------------------- LOAD MODELS -------------------------------------------------
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# ------------------------- UTILITY FUNCTIONS -------------------------------------------------
def weather_fetch(city_name):
    """Fetch weather data from OpenWeatherMap API"""
    api_key = weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}&units=metric"

    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()

        if "main" in data:
            return (
                round(data['main']['temp'], 2),
                data['main']['humidity'],
                data['weather'][0]['main'] if data.get('weather') else 'N/A'
            )
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Weather API Error: {str(e)}")
        return None

# ------------------------- ROUTES -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', title='Harvestify - Home')

@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='Harvestify - Crop Recommendation')

@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='Harvestify - Fertilizer Suggestion')

@app.route('/disease-check', methods=['GET', 'POST'])
def disease_check():
    """Handle both weather-based disease alerts and weather result display"""
    title = 'Harvestify - Disease Alerts'
    if request.method == 'POST':
        city = request.form.get('city')
        if not city:
            return render_template('disease.html', error="Please enter a city name", title=title)
        
        weather_data = weather_fetch(city)
        if not weather_data:
            return render_template('disease.html', error="Could not fetch weather data", title=title)
        
        temperature, humidity, conditions = weather_data
        alerts = get_disease_alerts(temperature, humidity)
        
        return render_template(
            'disease.html',
            title=title,
            weather={
                'city': city,
                'temperature': temperature,
                'humidity': humidity,
                'conditions': conditions
            },
            alerts=alerts
        )
    
    return render_template('disease.html', title=title)

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        weather_data = weather_fetch(city)
        if weather_data:
            temperature, humidity, _ = weather_data
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            final_prediction = crop_recommendation_model.predict(data)[0]
            return render_template(
                'crop-result.html',
                prediction=final_prediction,
                temperature=temperature,
                humidity=humidity,
                city=city,
                title=title
            )
        else:
            return render_template('try_again.html', title=title)

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'
    try:
        crop_name = str(request.form['cropname']).strip().lower()
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        fertilizer_csv_path = os.path.join(os.path.dirname(__file__), 'Data', 'fertilizer.csv')
        df = pd.read_csv(fertilizer_csv_path)
        df['Crop'] = df['Crop'].str.strip().str.lower()

        if crop_name not in df['Crop'].values:
            return render_template('fertilizer-result.html',
                                   recommendation=f"Crop '{crop_name}' not found in database.",
                                   title=title)

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n_diff = nr - N
        p_diff = pr - P
        k_diff = kr - K

        max_diff = max(abs(n_diff), abs(p_diff), abs(k_diff))
        if abs(n_diff) == max_diff:
            key = 'NHigh' if n_diff < 0 else 'Nlow'
        elif abs(p_diff) == max_diff:
            key = 'PHigh' if p_diff < 0 else 'Plow'
        else:
            key = 'KHigh' if k_diff < 0 else 'Klow'

        recommendation = Markup(str(fertilizer_dic[key]))
        return render_template('fertilizer-result.html',
                               recommendation=recommendation,
                               title=title)

    except Exception as e:
        return f"🔥 Internal Server Error: <pre>{str(e)}</pre>"

# ------------------------- MAIN -------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)