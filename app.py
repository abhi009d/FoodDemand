from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
import sys

app = Flask(__name__)

# Home Page
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# Prediction Route
@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    try:
        # Get form input values
        date = request.form['date']
        day_of_week = request.form['day_of_week']
        time_of_day = request.form['time_of_day']
        food_item = request.form['food_item']
        temperature = float(request.form['temperature'])
        rainfall_mm = float(request.form['rainfall_mm'])
        holiday = request.form['holiday']
        local_event = request.form['local_event']
        promotion_active = request.form['promotion_active']
        restaurant_location = request.form['restaurant_location']
        customer_footfall = int(request.form['customer_footfall'])

        # Prepare data for prediction
        data = CustomData(
            date=date,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
            food_item=food_item,
            temperature=temperature,
            rainfall_mm=rainfall_mm,
            holiday=holiday,
            local_event=local_event,
            promotion_active=promotion_active,
            restaurant_location=restaurant_location,
            customer_footfall=customer_footfall
        )

        final_df = data.get_data_as_data_frame()

        # Run prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(final_df)

        # Display result
        return render_template('home.html', results=f"📦 Predicted Orders: {round(prediction[0])}")

    except Exception as e:
        raise CustomException(e, sys)

# Run Flask app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

