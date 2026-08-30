import joblib
import pandas as pd

# Load trained model
model = joblib.load("telecom_tower_model.pkl")

# New tower data
new_data = pd.DataFrame({
    "Temperature_C": [55],
    "Battery_Voltage": [11.2],
    "Power_Consumption_W": [850],
    "Signal_Strength_Percent": [45],
    "Fan_Speed_RPM": [1800],
    "Humidity_Percent": [75],
    "Traffic_Load": [88],
    "Tower_Age_Years": [8]
})

# Make prediction
prediction = model.predict(new_data)

# Display result
if prediction[0] == 1:
    print("Hardware Failure Predicted")
else:
    print("Tower is Healthy")
