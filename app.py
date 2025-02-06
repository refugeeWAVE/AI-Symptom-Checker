from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
try:
    model = joblib.load("model/symptoms-disease-model.pkl")
except FileNotFoundError:
    print("Error: Model file not found. Make sure 'model/symptom_model.pkl' exists.")
    exit(1)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the AI Symptom Checker API! Use the /predict endpoint."

# ✅ Check if this route exists
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("symptoms", [])
        prediction = model.predict([data])[0]
        return jsonify({"disease_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
