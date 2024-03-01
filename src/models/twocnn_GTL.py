import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch import nn

class twocnn_GTL(LightningModule):  # McMahan et al., 2016; 1,663,370 parameters
    def __init__(self, in_channels, hidden_size, num_classes):
        super(twocnn_GTL, self).__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes
        self.activation = torch.nn.ReLU(True)



        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=(5, 5),
                            padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels * 2, kernel_size=(5, 5),
                            padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=(self.hidden_channels * 2) * (7 * 7), out_features=512, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        correct = sum(a.argmax() == b.argmax() for a, b in zip(y_hat, y))
        total = y.shape[0]
        return {"loss": loss, "correct": correct, "total": total}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        # implement your own
        out = self(x)
        loss = F.cross_entropy(out, y)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs!
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = sum(x["correct"] for x in outputs) / sum(x["total"] for x in outputs)
        self.log("Loss/Epoch", avg_loss, on_epoch=True, prog_bar=True)
        self.log("Acc/Epoch", avg_acc, on_epoch=True, prog_bar=True)