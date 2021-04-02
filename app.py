"""
Entry point for Flask app. Contain url path routing.
"""
import argparse
import json
import joblib
from multiprocessing import Pipe
import os

from flask import Flask, jsonify, request
from model.rnn import LstmModel
from model.utils import create_queue, create_thread, TaskType

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_route():
    """
    handles method post with json request {"x":[[],..]} to predict y
    :return: {"y":[...]}
    """
    if request.method == 'POST':
        if app.model is None:  # no model
            return 'First train the model!', 501
        try:
            X = request.json['x']
        except KeyError:
            return 'Need json {"x":[[],..]}', 400  # wrong json format

        child_pipe, parent_pipe = Pipe()  # pipe for receiving data from queue
        # send task wing args to queue
        app.rnn_task_queue.put((TaskType.PREDICT, [child_pipe, X, app.preprocessor, app.model, app.gpus]))
        y_pred = parent_pipe.recv()
        if isinstance(y_pred, ValueError):
            return str(y_pred), 400
        return jsonify({'y': y_pred.reshape(-1).tolist()})


@app.route('/train', methods=['POST'])
def train_route():
    """
    handles method post with json request {"x":[[],..], "y":[...]} to train new model
    :return:
    """
    if request.method == 'POST':
        try:
            X, y = request.json['x'], request.json['y']
        except KeyError:
            return 'Need json {"x":[[],..], "y":[...]}', 400

        child_pipe, parent_pipe = Pipe()
        app.rnn_task_queue.put((TaskType.TRAIN, [child_pipe, X, y, app.hparams, app.gpus]))
        state_dict, hparams, preprocessor, loss = parent_pipe.recv()
        if isinstance(state_dict, ValueError):
            return str(state_dict), 400
        elif state_dict is None:
            return 'Train error', 500
        else:
            app.model.load_state_dict(state_dict=state_dict)
            app.model.hparams = hparams
            app.preprocessor = preprocessor
            return jsonify({'loss': loss})


def get_args():
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join('weights', 'base_rnn.ckpt'),
                        help="Path RNN model weights")
    parser.add_argument('--normalizer_path', type=str, default=os.path.join('weights', f'data_normalizer.pkl'),
                        help="Path to sklearn Normalizer")
    parser.add_argument('--debug', type=boolean_string, default=True, help="Use debug mode")
    parser.add_argument('--port', type=int, default=5000, help="The port of the webserver. Defaults to `5000`")
    parser.add_argument('--use_gpu', type=boolean_string, default=True, help="Use gpu for training and inference")
    parser.add_argument('--hparams', type=str, default=os.path.join('weights', 'rnn_hparams.json'),
                        help="Path to model hparams")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if os.path.isfile(args.model_path) and os.path.isfile(args.normalizer_path):  # load model if exist
        app.model = LstmModel.load_from_checkpoint(checkpoint_path=args.model_path)
        app.preprocessor = joblib.load(args.normalizer_path)
    else:
        app.model = None
        app.preprocessor = None
    try:
        app.hparams = json.load(open(args.hparams))
    except FileNotFoundError:
        print("To start app need hparams file")
    app.rnn_task_queue = create_queue()  # using queue to avoid predict/train conflicts and gpu out of memory error
    app.queue_thread = create_thread(app.rnn_task_queue)  # thread to process queue
    app.queue_thread.start()
    app.gpus = args.use_gpu

    app.run(debug=args.debug, port=args.port)