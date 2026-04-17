import React, { useState } from "react";
import axios from "axios";

function Dashboard() {
  const [accuracy, setAccuracy] = useState(null);

  const train = async () => {
    const res = await axios.post("http://127.0.0.1:5000/train-model");
    setAccuracy(res.data.final_accuracy);
  };

  const predict = async () => {
    const res = await axios.post("http://127.0.0.1:5000/predict");
    alert("Prediction: " + res.data.prediction);
  };

  const federated = async () => {
    const res = await axios.post("http://127.0.0.1:5000/federated-round");
    alert(res.data.message);
  };

  return (
    <div>
      <h2>Dashboard</h2>
      <button onClick={train}>Train Model</button>
      <button onClick={predict}>Predict</button>
      <button onClick={federated}>Federated Learning</button>

      {accuracy && <h3>Accuracy: {accuracy}</h3>}

      <img src="http://127.0.0.1:5000/metrics.png" alt="Metrics" />
    </div>
  );
}

export default Dashboard;