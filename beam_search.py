from math import exp
from queue import PriorityQueue, Queue
from typing import Tuple

import torch
from ptgnn.neuralmodels.gnn import GnnOutput

from model.varNamingModel import VarNamingGraphModel


# Based on https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
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

            decoder_input = torch.cat([inp[-1:, :] for inp in decoder_inputs], dim=-1).to(var_rep.device)
            decoder_h = torch.cat(decoder_hs, dim=1).to(var_rep.device)

            # decode batch
            logits, hs = nn._decoder(decoder_input, decoder_h)
            for i, (token_idxs, log_p) in enumerate(zip(decoder_inputs, log_ps)):
                lp = torch.log_softmax(logits[0, i, :], -1)
                h = hs[:, i:i + 1, :]
                new_log_ps, new_idxs = torch.topk(lp, beam_width)
                for new_log_p, new_idx in zip(new_log_ps, new_idxs):
                    node = BeamSearchNode(h,
                                          torch.cat((token_idxs, new_idx.view(1, 1))),
                                          log_p + new_log_p)
                    nodes.put((-node.eval(), node))
                    qsize += 1
    return sorted(endnodes)


class BeamSearchNode(object):
    def __init__(self, hidden_state, token_idxs, log_probability):
        """
        :param token_idxs:
        :param log_probability:
        """
        self.hidden_state = hidden_state
        self.token_idxs = token_idxs
        self.log_p = log_probability

    def eval(self):
        length = self.token_idxs.shape[0]
        return self.log_p / float(length - 1 + 1e-8)
        # return self.log_p

    def __str__(self):
        return f"{self.token_idxs_array}: {self.p}"

    def __repr__(self):
        return str(self)

    @property
    def p(self):
        return exp(self.log_p)

    @property
    def token_idxs_array(self):
        return list(self.token_idxs.reshape(-1).cpu().numpy())

    def __lt__(self, other):
        return self.log_p < other.log_p