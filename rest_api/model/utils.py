"""
Contains queue and predict / train methods
"""
from datetime import datetime
from enum import Enum
import joblib
import numpy as np
import os
from threading import Thread
from queue import Queue

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import Normalizer
from sklearn.utils.validation import check_array

from model.rnn import TimeseriesDataset, train_pipeline, predict_pipeline


class TaskType(Enum):
    """
    Tasks enum class
    """
    TRAIN = 1
    PREDICT = 2


def create_queue():
    """
    create new queue
    :return: Queue
    """
    return Queue()


def create_thread(q: Queue):
    """
    New thread to process queue
    :param q: Queue
    :return: Thread
    """
    return Thread(target=process_rnn_queue, args=(q,))


def process_rnn_queue(q: Queue):
    """
    Method to process queue with torch model tasks.
    using queue to avoid predict/train conflicts and gpu out of memory error.
    To send results use pipes
    :param q: Queue
    :return: None
    """
    while True:
        # get task form queue
        task, task_args = q.get()
        print(task, task_args)
        if task == TaskType.TRAIN:
            [child_pipe, X, y, hparams, gpus] = task_args
            try:
                model, preprocessor, loss = train(X=X, y=y, hparams=hparams, gpus=gpus)
                state_dict = model.state_dict()  # cant send model via pipe, need to use state dict
                hparams = model.hparams
                child_pipe.send([state_dict, hparams, preprocessor, loss])
            except ValueError as e:
                child_pipe.send([e, None, None, None])  # incorrect data format
            except Exception:
                child_pipe.send([None, None, None, None])  # smt happened
        elif task == TaskType.PREDICT:
            [child_pipe, X, preprocessor, model, gpus] = task_args
            try:
                y_pred = predict(X=X, preprocessor=preprocessor, model=model, gpus=gpus)
                child_pipe.send(y_pred)
            except ValueError as e:
                child_pipe.send(e)  # incorrect data format
            except Exception:
                child_pipe.send(None)  # smt happened


def predict(X, preprocessor, model, gpus=None):
    """
    Preprocess data.  Predict y by X input.
    :param X:            - array-like of shape (n_samples, n_features)
    :param preprocessor: - sklearn.preprocessing object
    :param model:        - LightningModule model
    :param batch_size:   - num samples in batch
    :param gpus:         - num gpus for model
    :return: y_pred      - array-like of shape (n_samples, 1)

    """
    gpus = check_gpu(gpus)
    device = 'cpu' if gpus == 0 else "cuda"
    seq_len = model.hparams.seq_len
    X = check_arrays(X)
    if X.shape[1] != model.hparams.n_features:
        raise ValueError(f"Wrong input features. Requires {model.n_features}, but passed {X.shape[1]}")
    X = preprocessor.transform(X)
    seq_x = [X[max(0, i - seq_len + 1): i + 1] for i in range(X.shape[0])]  # create sequences from X
    y_pred = predict_pipeline(seq_x=seq_x, model=model, device=device)
    return y_pred


def train(X, y, hparams, gpus=None):
    """
    Preprocess data. Train new LstmModel and preprocessor
    :param X:        - array-like of shape (n_samples, n_features)
    :param y:        - array-like of shape (n_samples, 1)
    :param hparams:  - model hparams {"n_features":, "hidden_layer_size":, "num_layers":, "seq_len":,
                                      "learning_rate": "dropout": "weight_decay":}
    :param gpus:     - num gpus for model
    :return: model, preprocessor, loss
    """
    gpus = check_gpu(gpus)
    X, y = check_arrays(X, y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Wrong input features. Requires X.shape[0] == y.shape[0], "
                         f"passed {X.shape[0]} and {y.shape[0]}")
    if X.shape[0] < hparams["seq_len"]:
        raise ValueError(f"Min number of samples must be > {hparams['seq_len']}, but passed {X.shape[0]} samples")
    hparams['n_features'] = X.shape[1]  # save new features num if changed
    preprocessor = Normalizer()
    X = preprocessor.fit_transform(X)

    train_loader = DataLoader(TimeseriesDataset(X, y, hparams["seq_len"]),
                              batch_size=hparams['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(TimeseriesDataset(X, y, hparams["seq_len"]),
                            batch_size=1000, shuffle=False, num_workers=0)

    # train model
    trainer = train_pipeline(train_loader=train_loader, val_loader=val_loader, params=hparams, gpus=gpus)
    trainer.test(test_dataloaders=val_loader, verbose=False)  # compute mape loss
    loss = trainer.model.test_result
    model = trainer.model.to('cpu')

    # save model and preprocessor
    time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join('weights', f'rnn_{time_now}.ckpt')
    normalizer_path = os.path.join('weights', f'data_normalizer_{time_now}.pkl')

    joblib.dump(preprocessor, normalizer_path)  # save sklearn.Normalizer
    trainer.save_checkpoint(model_path)  # save model weights
    return model, preprocessor, loss


def check_gpu(gpus):
    """
    Check or set gpu
    :param gpus: number of gpu or None
    :return: number of gpu
    """
    if gpus is None:
        gpus = int(torch.cuda.is_available())
    elif not torch.cuda.is_available():
        gpus = None
    return int(gpus)


def check_arrays(X, y=None):
    """
    Check arrays and convert to np.array
    :param X: - array-like
    :param y: - array-like
    :return: X, y if not None
    """
    try:
        X = np.array(check_array(X))
        if y is not None:
            y = np.array(check_array(y, ensure_2d=False))  # check input
            return X, y
        else:
            return X
    except ValueError:
        raise ValueError("Broken input")


