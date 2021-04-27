import json
import logging
import time
import warnings
from collections import namedtuple
from os.path import join
from pathlib import Path

import torch
from flask import Flask, request
from flask_ngrok import run_with_ngrok
from omegaconf import OmegaConf

from beam_search import beam_search
from model.varNamingModel import VarNamingModel, VarNamingSample

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
run_with_ngrok(app)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: ", DEVICE)
warnings.filterwarnings("ignore")

CONFIGS_DIR = r"/content/drive/MyDrive/Id Names Suggesting/configs/gnn"
# CONFIGS_DIR = r"C:\Users\Igor\PycharmProjects\id-gnn\configs"


def load_config(configs_path=r'transformer/configs/'):
    cfg = OmegaConf.load(join(configs_path, 'config.yaml'))
    cfg['model'] = OmegaConf.load(join(configs_path, r'model/ggnn.yaml'))
    cfg['dataset'] = OmegaConf.load(join(configs_path, r'dataset/java-small.yaml'))
    return cfg


cfg = load_config(CONFIGS_DIR)
restore_path = cfg.model.restore_path
if restore_path:
    model, nn = VarNamingModel.restore_model(Path(restore_path), DEVICE)
    nn.train(False)
else:
    ValueError("Specify restore_path!")

LastInference = namedtuple("LastInference", ["predictions", "gt", "time_spent"])
last_inference = LastInference(None, None, None)


@app.route('/', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        start = time.perf_counter()
        try:
            var_sample: VarNamingSample = request.get_json(force=True, cache=False)
            gt = var_sample["name"]
            tensorized_sample = model.tensorize(var_sample)

            minibatch = model.initialize_minibatch()
            model.extend_minibatch_with(tensorized_sample, minibatch)
            minibatch = model.finalize_minibatch(minibatch, device=DEVICE)
            minibatch["target"]["token_idxs"] = minibatch["target"]["token_idxs"][:1]
            predictions = beam_search(nn, **minibatch)
            predictions = list(map(lambda prediction:
                                   {
                                       "name": [model.vocabulary.id_to_token[
                                                    token_idx] for token_idx in prediction[1].token_idxs.squeeze()][1: -1],
                                       "p": prediction[1].p
                                   }, predictions))
        except Exception:
            return
        time_spent = time.perf_counter() - start
        global last_inference
        last_inference = LastInference(predictions, gt, time_spent)
        return json.dumps({"predictions": last_inference.predictions,
                           "gnnEvaluationTime": last_inference.time_spent})


@app.route('/')
def running():
    global last_inference
    return "I don't have anything to show!" if last_inference.gt is None \
        else f"Running!<p>Last inference:<p>Time spent: {last_inference.time_spent}<p>" + \
             f"<p>Ground truth: {last_inference.gt}<p>" + \
             f"Predictions: {last_inference.predictions}"


app.run()
