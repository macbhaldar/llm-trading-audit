import torch

from torch.utils.data import DataLoader


class LSTMTrainer:

    def __init__(

        self,

        model,

        config,

        dataset

    ):

        self.model = model

        self.config = config

        self.loader = DataLoader(

            dataset,

            batch_size=config.batch_size,

            shuffle=True
        )

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=config.learning_rate
        )

    def train(self):

        self.model.train()

        for epoch in range(self.config.epochs):

            epoch_loss = 0

            for X, y in self.loader:

                pred = self.model(X)

                loss = self.loss_fn(pred, y)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

            print(

                f"Epoch "

                f"{epoch+1}/"

                f"{self.config.epochs}"

                f" Loss: "

                f"{epoch_loss/len(self.loader):.6f}"
            )