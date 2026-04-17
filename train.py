import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
from model import GNNModel
from plot import plot_metrics

def create_graph():
    x = torch.tensor([
        [65, 1, 120],
        [45, 0, 130],
        [50, 1, 140],
        [30, 0, 110],
        [70, 1, 150]
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 0],
        [1, 0, 3, 2, 0, 4]
    ], dtype=torch.long)

    y = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def train_model():
    data = create_graph()
    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    accuracies = []

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = accuracy_score(data.y.numpy(), pred.detach().numpy())

        losses.append(loss.item())
        accuracies.append(acc)

    plot_metrics(losses, accuracies)

    return model, losses, accuracies