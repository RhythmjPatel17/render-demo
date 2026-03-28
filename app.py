# ================================
# Step 1: Import Libraries
# ================================
from flask import Flask, render_template_string, request
import joblib
import pandas as pd

# ================================
# Step 2: Load Models
# ================================
car_artifact = joblib.load("car_models.pkl")
laptop_artifact = joblib.load("laptop_model.pkl")

car_model = car_artifact["decision_tree_gini"]
car_encoders = car_artifact["feature_encoders"]
car_target_encoder = car_artifact["target_encoder"]   # ✅ FIX ADDED

laptop_model = laptop_artifact["model"]
laptop_encoders = laptop_artifact["feature_encoders"]
laptop_columns = laptop_artifact["feature_columns"]

# ================================
# Step 3: Initialize App
# ================================
app = Flask(__name__)

# ================================
# Step 4: HTML Template
# ================================
template = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Prediction App</title>

    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI';
            background: linear-gradient(-45deg, #1f4037, #99f2c8, #ff7e5f, #6a11cb);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
            color: white;
            overflow-x: hidden;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .floating {
            position: absolute;
            font-size: 40px;
            animation: float 6s ease-in-out infinite;
            opacity: 0.7;
        }

        .car { top: 10%; left: 5%; animation-delay: 0s; }
        .laptop { bottom: 10%; right: 5%; animation-delay: 2s; }

        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0px); }
        }

        .container {
            text-align: center;
            margin-top: 50px;
        }

        .box {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            display: inline-block;
            margin: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            transition: 0.3s;
        }

        .box:hover {
            transform: scale(1.05);
        }

        input {
            padding: 10px;
            margin: 6px;
            border-radius: 6px;
            border: none;
            width: 200px;
        }

        button {
            padding: 10px 20px;
            margin-top: 10px;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            border: none;
            color: white;
            border-radius: 20px;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            transform: scale(1.1);
        }

        h1 {
            font-size: 40px;
        }

        h3 {
            margin-top: 15px;
            color: #fff;
        }
    </style>
</head>

<body>

<div class="floating car">🚗</div>
<div class="floating laptop">💻</div>

<div class="container">
    <h1>🚀 Smart ML Prediction System</h1>

    <div class="box">
        <h2>Select Model</h2>
        <form method="POST">
            <button name="mode" value="car">🚗 Car Evaluation</button>
            <button name="mode" value="laptop">💻 Laptop Price</button>
        </form>
    </div>

    {% if mode == "car" %}
    <div class="box">
        <h2>🚗 Car Prediction</h2>
        <form method="POST">
            <input name="buying" placeholder="buying"><br>
            <input name="maint" placeholder="maint"><br>
            <input name="doors" placeholder="doors"><br>
            <input name="persons" placeholder="persons"><br>
            <input name="lug_boot" placeholder="lug_boot"><br>
            <input name="safety" placeholder="safety"><br>
            <button type="submit" name="predict_car">Predict</button>
        </form>
        <h3>{{ result }}</h3>
    </div>
    {% endif %}

    {% if mode == "laptop" %}
    <div class="box">
        <h2>💻 Laptop Price Prediction</h2>
        <form method="POST">
            <input name="Company" placeholder="Company"><br>
            <input name="TypeName" placeholder="TypeName"><br>
            <input name="Ram" placeholder="Ram"><br>
            <input name="OpSys" placeholder="OpSys"><br>
            <input name="Weight" placeholder="Weight"><br>
            <input name="Touchscreen" placeholder="Touchscreen"><br>
            <input name="CpuCompany" placeholder="CpuCompany"><br>
            <input name="ClockSpeed" placeholder="ClockSpeed"><br>
            <input name="Flash Storage" placeholder="Flash Storage"><br>
            <input name="HDD" placeholder="HDD"><br>
            <input name="Hybrid" placeholder="Hybrid"><br>
            <input name="SSD" placeholder="SSD"><br>
            <input name="GPU" placeholder="GPU"><br>
            <input name="Ppi" placeholder="Ppi"><br>
            <input name="Price" placeholder="Price"><br>
            <button type="submit" name="predict_laptop">Predict</button>
        </form>
        <h3>{{ result }}</h3>
    </div>
    {% endif %}

</div>

</body>
</html>
"""

# ================================
# Step 5: Main Route
# ================================
@app.route('/', methods=['GET', 'POST'])
def home():
    mode = None
    result = ""

    if request.method == 'POST':

        # Select Mode
        if 'mode' in request.form:
            mode = request.form['mode']

        # ================================
        # Car Prediction
        # ================================
        elif 'predict_car' in request.form:
            mode = "car"

            data = {key: [request.form[key]] for key in ['buying','maint','doors','persons','lug_boot','safety']}
            df = pd.DataFrame(data)

            try:
                # SAFE encoding
                for col, encoder in car_encoders.items():
                    if col in df.columns:
                        df[col] = encoder.transform(df[col])

                pred = car_model.predict(df)[0]

                # Decode output label
                pred_label = car_target_encoder.inverse_transform([pred])[0]

                result = f"Prediction: {pred_label}"

            except:
                result = "Invalid input! Please check values."

        # ================================
        # Laptop Prediction
        # ================================
        elif 'predict_laptop' in request.form:
            mode = "laptop"

            data = {key: [request.form[key]] for key in request.form if key != 'predict_laptop'}
            df = pd.DataFrame(data)

            try:
                # Convert numeric values
                for col in df.columns:
                    try:
                        df[col] = df[col].astype(float)
                    except:
                        pass

                # Encode categorical
                for col, encoder in laptop_encoders.items():
                    df[col] = encoder.transform(df[col])

                # Feature Engineering
                df['Total_Storage'] = df['SSD'] + df['HDD'] + df['Flash Storage']
                df['Price_per_RAM'] = df['Price'] / (df['Ram'] + 1)

                df = df[laptop_columns]

                pred = laptop_model.predict(df)[0]
                result = f"Predicted Price: ₹{round(pred,2)}"

            except:
                result = "Invalid input! Please check values."

    return render_template_string(template, mode=mode, result=result)


# ================================
# Step 6: Run App
# ================================
if __name__ == '__main__':
    app.run(debug=True)
