import torch

from torch.utils.data import DataLoader


class TransformerTrainer:

    def __init__(

        self,

        model,

        config,

        train_dataset,

    ):

        self.model = model

        self.config = config

        self.loader = DataLoader(

            train_dataset,

            batch_size=config.batch_size,

            shuffle=True

        )

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(

            model.parameters(),

            lr=config.learning_rate

        )

    def train(self):

        self.model.train()

        for epoch in range(self.config.epochs):

            loss_sum = 0

            for X, y in self.loader:

                pred = self.model(X)

                loss = self.loss_fn(

                    pred.squeeze(),

                    y

                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                loss_sum += loss.item()

            print(

                f"Epoch {epoch+1}: "

                f"{loss_sum/len(self.loader):.6f}"

            )