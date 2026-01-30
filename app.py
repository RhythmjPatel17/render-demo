from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Load the trained model and encoders
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    artifact = pickle.load(f)

model = artifact["random_forest"]
feature_encoders = artifact["feature_encoders"]
target_encoder = artifact["target_encoder"]

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input in the correct feature order
        input_data = [
            request.form["buying"],
            request.form["maint"],
            request.form["doors"],
            request.form["persons"],
            request.form["lug_boot"],
            request.form["safety"]
        ]

        # Encode categorical features
        encoded_input = []
        for i, col in enumerate(feature_encoders.keys()):
            encoded_val = feature_encoders[col].transform([input_data[i]])[0]
            encoded_input.append(encoded_val)

        final_input = np.array(encoded_input).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_input)
        result = target_encoder.inverse_transform(prediction)[0]

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        # Catch and display any error
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)