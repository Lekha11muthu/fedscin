import torch
from train import train_model

def federated_training(rounds=3):
    global_model = None

    for r in range(rounds):
        local_models = []

        for client in range(3):
            model, _, _ = train_model()
            local_models.append(model)

        global_model = local_models[0]

        for key in global_model.state_dict().keys():
            global_model.state_dict()[key].data.copy_(
                torch.mean(torch.stack([
                    m.state_dict()[key].float() for m in local_models
                ]), dim=0)
            )

    return global_model