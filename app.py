from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("telecom_tower_model.pkl")


@app.route("/")
def home():
    return "Telecom Tower Prediction API is Running Successfully!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input data into DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)

    return jsonify({
        "prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)
