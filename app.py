from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and encoder
model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le_sex = joblib.load("model/sex_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            pclass = int(request.form["pclass"])
            sex = request.form["sex"]
            age = float(request.form["age"])
            sibsp = int(request.form["sibsp"])
            fare = float(request.form["fare"])

            sex_encoded = le_sex.transform([sex])[0]

            input_data = np.array([[pclass, sex_encoded, age, sibsp, fare]])
            input_scaled = scaler.transform(input_data)
            result = model.predict(input_scaled)[0]

            prediction = "Survived" if result == 1 else "Did Not Survive"

        except:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
