## Description 

Flask REST service to train and inference RNN model.

## Installation

Python requirements:

`pip install -r requirements.txt `

For best performance it is highly  recommended to install [CUDA](https://developer.nvidia.com/cuda-downloads?).

## Usage

To use pretrained models, download from [drive](https://drive.google.com/drive/folders/1urhOsF4LcZsqeZXkDAVCbTqu6gDyOVmn) `base_rnn.ckpt`, `data_normalizer.pkl` and paste them to `weights` folder.

```sh
python app.py -h
    usage: app.py [-h] [--model_path MODEL_PATH]
                  [--normalizer_path NORMALIZER_PATH] [--debug DEBUG]
                  [--port PORT] [--use_gpu USE_GPU] [--hparams HPARAMS]

    optional arguments:
      -h, --help            show this help message and exit
      --model_path MODEL_PATH
                            Path RNN model weights (default:
                            weights\base_rnn.ckpt)
      --normalizer_path NORMALIZER_PATH
                            Path to sklearn Normalizer (default:
                            weights\data_normalizer.pkl)
      --debug DEBUG         Use debug mode (default: True)
      --port PORT           The port of the webserver. Defaults to `5000`
                            (default: 5000)
      --use_gpu USE_GPU     Use gpu for training and inference (default: True)
      --hparams HPARAMS     Path to model hparams (default:
                            weights\rnn_hparams.json)


```

To predict: POST json  on path `/predict`. 

```json
request json example:
{"x": [
    [1,2,...],
    [..],
    .
]}
    
response json example:
{"y": [
    1,
    2,
    .
]}  
```

To train: POST json  on path `/train`. 

```json
request json example:
{"x": [
    [1,2,...],
    [..],
    .
],
 "y":[
    1,
    2,
    .
]}
    
response json example:
{"loss": 1} # mape%  
```

More examples in Postman collection `rest_rnn_service.postman_collection.json`