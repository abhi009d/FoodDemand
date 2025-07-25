import gradio as gr
import pandas as pd
import dill

from src.utils import load_object  # or use dill.load(open(...)) directly

# Load model and preprocessor
model = load_object("artifacts/model.pkl")
preprocessor = load_object("artifacts/preprocessor.pkl")

def predict_demand(date, day_of_week, time_of_day, food_item, temperature, rainfall_mm,
                   holiday, local_event, promotion_active, restaurant_location, customer_footfall):

    # Build input dataframe
    data = pd.DataFrame([{
        "date": date,
        "day_of_week": day_of_week,
        "time_of_day": time_of_day,
        "food_item": food_item,
        "temperature": float(temperature),
        "rainfall_mm": float(rainfall_mm),
        "holiday": holiday,
        "local_event": local_event,
        "promotion_active": promotion_active,
        "restaurant_location": restaurant_location,
        "customer_footfall": int(customer_footfall)
    }])

    # Preprocess
    data_transformed = preprocessor.transform(data)

    # Predict
    prediction = model.predict(data_transformed)
    return f"🔮 Predicted Orders: {int(prediction[0])}"

# Gradio Interface
demo = gr.Interface(
    fn=predict_demand,
    inputs=[
        gr.Textbox(label="Date (YYYY-MM-DD)"),
        gr.Textbox(label="Day of Week"),
        gr.Textbox(label="Time of Day"),
        gr.Textbox(label="Food Item"),
        gr.Number(label="Temperature"),
        gr.Number(label="Rainfall (mm)"),
        gr.Radio(["Yes", "No"], label="Holiday"),
        gr.Radio(["Yes", "No"], label="Local Event"),
        gr.Radio(["Yes", "No"], label="Promotion Active"),
        gr.Textbox(label="Restaurant Location"),
        gr.Number(label="Customer Footfall")
    ],
    outputs="text",
    title="🍽️ Restaurant Food Demand Predictor"
)

demo.launch()
