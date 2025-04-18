# api.py
from flask import Flask, request, jsonify
from model import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def make_prediction():
    data = request.json
    features = data.get("features")
    if not features:
        return jsonify({"error": "Missing 'features' in request"}), 400

    try:
        result = predict(features)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
