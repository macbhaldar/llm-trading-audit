import torch
import numpy as np


class LSTMPredictor:

    def __init__(self, model):

        self.model = model

        self.model.eval()

    @torch.no_grad()

    def predict(self, X):

        X = torch.tensor(

            X,

            dtype=torch.float32
        )

        prediction = self.model(X)

        return prediction.cpu().numpy()

    