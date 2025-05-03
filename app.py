from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from HTML frontend

# Sample training data: Last 3 colors → Next color
# Red = 0, Green = 1
X = [
    [0, 0, 0], [1, 1, 1], [0, 1, 0],
    [1, 0, 1], [0, 0, 1], [1, 1, 0],
    [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]
]
y = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # Next color

# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X, y)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    history = data.get("history", [])

    if len(history) < 3:
        return jsonify({"error": "Not enough data"}), 400

    last3 = history[-3:]
    input_data = [[0 if c == "Red" else 1 for c in last3]]
    pred = clf.predict(input_data)[0]
    prob = clf.predict_proba(input_data)[0][pred]

    return jsonify({
        "prediction": "Red" if pred == 0 else "Green",
        "confidence": round(prob * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
