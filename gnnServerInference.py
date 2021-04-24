import json
import logging
import time
import warnings
from collections import namedtuple
from os.path import join
from pathlib import Path
from queue import PriorityQueue, Queue
from typing import Tuple

import torch
from flask import Flask, request
from flask_ngrok import run_with_ngrok
from omegaconf import OmegaConf
from ptgnn.neuralmodels.gnn import GnnOutput

from model.varNamingModel import VarNamingModel, VarNamingSample, VarNamingGraphModel
from utils import INIT_TOKEN, BeamSearchNode

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
run_with_ngrok(app)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: ", DEVICE)
warnings.filterwarnings("ignore")

# CONFIGS_DIR = r"/content/drive/MyDrive/Id Names Suggesting/transformer/configs"
CONFIGS_DIR = r"C:\Users\Igor\PycharmProjects\id-gnn\configs"


def load_config(configs_path=r'transformer/configs/'):
    cfg = OmegaConf.load(join(configs_path, 'config.yaml'))
    cfg['model'] = OmegaConf.load(join(configs_path, r'model/ggnn.yaml'))
    cfg['dataset'] = OmegaConf.load(join(configs_path, r'dataset/java-small.yaml'))
    return cfg


cfg = load_config(CONFIGS_DIR)
restore_path = cfg.model.restore_path
if restore_path:
    model, nn = VarNamingModel.restore_model(Path(restore_path), DEVICE)
else:
    ValueError("Specify restore_path!")

LastInference = namedtuple("LastInference", ["predictions", "gt", "time_spent"])
last_inference = LastInference(None, None, None)


def beam_search(nn: VarNamingGraphModel, graph_data, target,
                topk=10, batch_size=10, beam_width=40, max_output_length=6):
    with torch.no_grad():
        gnn_output: GnnOutput = nn._gnn(**graph_data)

        var_node_idxs = gnn_output.node_idx_references["var_node_idxs"]
        var_node_reps = gnn_output.output_node_representations[var_node_idxs]

        var_rep = var_node_reps.mean(0, keepdims=True)  # A [1, D] tensor, containing var embedding.

        # Start with the start of the sentence token
        init_token = target["token_idxs"]

        # Number of sentence to generate
        endnodes = []

        # starting node -  previous node, word id, logp, length
        node = BeamSearchNode(var_rep.unsqueeze(0), init_token, 0)
        nodes: Queue[Tuple[float, BeamSearchNode]] = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        # give up when decoding takes too long
        while qsize < 10000:
            # fetch batch of the best nodes
            decoder_inputs, decoder_hs, log_ps = [], [], []
            while qsize > 0 and len(decoder_inputs) < batch_size:
                score, n = nodes.get()
                qsize -= 1
                token_idxs = n.token_idxs
                if token_idxs.shape[0] >= max_output_length or token_idxs[-1, 0] == nn._decoder.eos_idx:
                    endnodes.append((score, n))
                    if len(endnodes) >= topk:
                        break
                    continue
                decoder_inputs.append(token_idxs)
                decoder_hs.append(n.hidden_state)
                log_ps.append(n.log_p)

            if len(endnodes) >= topk:
                break

            decoder_input = torch.cat([inp[-1:] for inp in decoder_inputs], dim=-1).to(var_rep.device)
            decoder_h = torch.cat(decoder_hs, dim=1).to(var_rep.device)

            # decode batch
            logits, hs = nn._decoder(decoder_input, decoder_h)
            for i, (token_idxs, log_p) in enumerate(zip(decoder_inputs, log_ps)):
                logit = torch.log_softmax(logits[0, i, :], -1)
                h = hs[:, i:i + 1, :]
                new_log_ps, new_idxs = torch.topk(logit, beam_width)
                for new_log_p, new_idx in zip(new_log_ps, new_idxs):
                    node = BeamSearchNode(h,
                                          torch.cat((token_idxs, new_idx.view(1, 1))),
                                          log_p + new_log_p)
                    score = -node.eval()
                    nodes.put((score, node))
                    qsize += 1
    return sorted(endnodes)


@app.route('/', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        start = time.perf_counter()
        var_sample: VarNamingSample = request.get_json(force=True, cache=False)
        gt = var_sample["name"]
        var_sample["name"] = INIT_TOKEN
        var_sample["types"] = []
        tensorized_sample = model.tensorize(var_sample)

        minibatch = model.initialize_minibatch()
        model.extend_minibatch_with(tensorized_sample, minibatch)
        minibatch = model.finalize_minibatch(minibatch, device=DEVICE)
        predictions = beam_search(nn, *minibatch)
        predictions = list(map(lambda prediction:
                               {
                                   "name": [model.__decoder_model.__target_embedding_model.vocabulary.id_to_token(
                                       token_idx) for token_idx in prediction[1].token_idxs.squeeze()][1: -1],
                                   "p": prediction[1].p
                               }, predictions))
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
