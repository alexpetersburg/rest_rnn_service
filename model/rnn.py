"""
Contains torch model and train / predict base methods
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


class LstmModel(LightningModule):
    """
    PytorchLightning model, more readable than default pytorch model
    """

    def __init__(self, n_features, hidden_layer_size=20, num_layers=3, learning_rate=0.01, dropout=0.3,
                 weight_decay=0.001, seq_len=10, batch_size=200):
        """
        n_features:         - number of dataset features
        hidden_layer_size:  - LSTM and linear layer size
        num_layers:         - number of LSTM layers
        learning_rate:      - initial learning rate
        dropout:            - LSTM dropout
        weight_decay:       - add L2 penalty
        seq_len:            - input sequence len
        """
        super().__init__()
        self.save_hyperparameters()  # save hparam in checkpoint
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        # x (batch size, seq_len, n_features)
        y_hat, _ = self.lstm(x)  # output, (h_n, c_n)
        y_hat = self.linear(y_hat[:, -1, :])  # pass only last lstm output
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat.view(-1), y)  # mse as loss
        self.log('train_loss', loss)  # log to tensorboard
        return {'loss': loss}  # return loss to optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,
                                                               factor=0.5, )  # drop lr if monitor
        # value not improving
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mape"}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mape = torch.abs(y - y_hat) / y  # val mape loss to lr moninor and earlystop moninor
        return {'val_loss': F.l1_loss(y_hat, y.view(-1, 1)), 'val_mape': mape}  # , 'val_mape': mape

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mape = torch.flatten(torch.cat([x['val_mape'].view(-1) for x in outputs])).mean()
        self.log('val_loss', avg_loss)  # log to tensorboard
        self.log('val_mape', mape)  # log to tensorboard

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mape = torch.abs(y - y_hat) / y.abs()
        return {'mape': mape}

    def test_epoch_end(self, outputs):
        outputs = torch.flatten(torch.cat([x['mape'].view(-1) for x in outputs]))
        mape = torch.sum(outputs) / outputs.size()[0]
        self.test_result = mape.item() * 100


class TimeseriesDataset(Dataset):
    """
    Custom Dataset.
    Return (seq_len, n_features).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 15):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])


def train_pipeline(train_loader, val_loader, params, gpus=None):
    """
    Train model using params
    :param train_loader: - torch.utils.data.DataLoader
    :param val_loader:   - torch.utils.data.DataLoader
    :param params:       - model hparams {"n_features":, "hidden_layer_size":, "num_layers":,
                                          "learning_rate": "dropout": "weight_decay":}
    :param gpus:         - num gpus for model
    :return:             - pytorch_lightning.Trainer
    """

    model = LstmModel(n_features=params['n_features'],
                      hidden_layer_size=params['hidden_layer_size'],
                      num_layers=params['num_layers'],
                      dropout=params['dropout'],
                      weight_decay=params['weight_decay'],
                      seq_len=params['seq_len'],
                      batch_size=params['batch_size'])
    trainer = Trainer(
        max_epochs=50,
        gpus=gpus,
        progress_bar_refresh_rate=0,
        logger=None,
        weights_summary=None,
        auto_lr_find=True,  # allows find initial lr automatically
        callbacks=[EarlyStopping(monitor='val_mape', patience=5)]  # stop if monitor value not improving
    )
    lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader, val_dataloaders=val_loader,
                                      num_training=100)
    new_lr = lr_finder.suggestion()
    model.learning_rate = new_lr
    model.hparams.learning_rate = new_lr  # save lr to hparams to save in checkpoint

    trainer.fit(model, train_loader, val_loader)  # training
    return trainer


def predict_pipeline(seq_x, model, device):
    """
    Predict samples
    :param seq_x:          - list of X sequences
    :param model:          - LightningModule model
    :param device:         - cpu or cuda
    :return:               - np.array
    """
    model.eval()
    outputs = []
    model.to(device)
    with torch.no_grad():
        for X in seq_x:
            X = torch.tensor(X).float().unsqueeze(0)
            X = X.to(device)
            y_pred = model(X)
            outputs.append(y_pred.view(-1).to('cpu'))
    return torch.flatten(torch.cat(outputs)).numpy()



