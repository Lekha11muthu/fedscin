from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from train import train_model
from federated import federated_training
from predict import predict
from torch_geometric.data import Data
import torch

app = Flask(__name__)
CORS(app)

global_model = None
import os

if os.path.exists("model.pth"):
    try:
        global_model, _, _ = train_model()  # initialize structure
        global_model.load_state_dict(torch.load("model.pth"))
        global_model.eval()
        print("Model loaded from file")
    except:
        print("Model load failed")

@app.route("/")
def home():
    return "FedSCIN AI Backend Running"

@app.route("/train-model", methods=["POST"])
def train():
    global global_model
    global_model, losses, accuracies = train_model()

    return jsonify({
        "message": "Model trained",
        "final_loss": losses[-1],
        "final_accuracy": accuracies[-1]
    })

@app.route("/metrics.png")
def get_metrics():
    return send_file("metrics.png", mimetype='image/png')

@app.route("/predict", methods=["POST"])
def make_prediction():
    global global_model

    if global_model is None:
        return jsonify({"error": "Train model first"})

    data_json = request.get_json()
    values = data_json["features"]   # dynamic input

    x = torch.tensor([values], dtype=torch.float)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    result = predict(global_model, data)

    label = "High Risk" if int(result[0]) == 1 else "Low Risk"

    return jsonify({
        "prediction": int(result[0]),
        "label": label
    })
if __name__ == "__main__":
    app.run(debug=True)