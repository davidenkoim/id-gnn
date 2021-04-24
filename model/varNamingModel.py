import re
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from dpu_utils.codeutils import split_identifier_into_parts
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.gnn.graphneuralnetwork import GraphNeuralNetwork, GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.structs import GnnOutput, GraphData, TensorizedGraphData
from torch import nn
from torch_scatter import scatter_mean
from typing_extensions import Final, TypedDict

from model.rnnDecoderModel import RNNDecoder, RNNDecoderModel, RNNOutput
from utils import VAR_TOKEN


class VarNamingGraph(TypedDict):
    Edges: Dict[str, List[Tuple[int, int]]]
    NodeLabels: Dict[str, str]


class VarNamingSample(TypedDict):
    ContextGraph: VarNamingGraph
    name: str
    types: List[str]


class TensorizedVarNamingSample(NamedTuple):
    graph: TensorizedGraphData
    target_idxs: List[int]


class VarNamingGraphModel(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork, decoder: RNNDecoder):
        super().__init__()
        self._gnn = gnn
        self._decoder = decoder
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self._ignore_idxs = (decoder.pad_idx, decoder.eos_idx)

    def _reset_module_metrics(self) -> None:
        self.__sum_acc = 0
        self.__num_samples = 0
        self.__tp = 0
        self.__fp = 0
        self.__fn = 0
        self.__sum_loss = 0
        self.__num_batches = 0

    def _module_metrics(self) -> Dict[str, Any]:
        loss_epoch = self.__sum_loss / self.__num_batches if self.__num_batches != 0 else 0
        accuracy = self.__sum_acc / self.__num_samples if self.__num_samples != 0 else 0
        precision = self.__tp / (self.__tp + self.__fp) if (self.__tp + self.__fp) != 0 else 0
        recall = self.__tp / (self.__tp + self.__fn) if (self.__tp + self.__fn) != 0 else 0
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0,
            "loss_epoch": loss_epoch
        }

    def forward(self, graph_data, target):
        gnn_output: GnnOutput = self._gnn(**graph_data)

        var_node_idxs = gnn_output.node_idx_references["var_node_idxs"]
        var_node_reps = gnn_output.output_node_representations[var_node_idxs]

        var_node_to_graph = gnn_output.node_graph_idx_reference["var_node_idxs"]

        var_reps = scatter_mean(
            src=var_node_reps,
            index=var_node_to_graph.unsqueeze(1),
            dim=0
        )  # A [num_graphs, D] tensor, containing one vector per graph.

        token_idxs = target["token_idxs"]
        rnn_output: RNNOutput = self._decoder(token_idxs, var_reps.unsqueeze(0))
        logits = rnn_output.logits[:-1]
        idxs = token_idxs[1:]
        loss = self._loss(logits.transpose(1, 2), idxs).sum(0).mean()
        with torch.no_grad():
            self._update_metrics(logits, idxs, loss.detach().cpu().item())
        return loss

    def _update_metrics(self, logits: torch.Tensor, token_idxs: torch.Tensor, loss) -> None:  # [L, B, V], [L, B]
        self.__sum_loss += loss
        self.__num_batches += 1
        if logits.numel() == 0:
            return
        pred_token_idxs = logits.argmax(-1)  # [L, B, V] -> [L, B]
        predictions = pred_token_idxs.transpose(0, 1)  # [L, B] -> [B, L]
        targets = token_idxs.transpose(0, 1)  # [L, B] -> [B, L]
        for prediction, target in zip(predictions, targets):
            self.__num_samples += 1
            self.__sum_acc += torch.eq(prediction, target).all().cpu().item()
            for pred_subtoken in filter(lambda x: x not in self._ignore_idxs, prediction):
                if pred_subtoken in target:
                    self.__tp += 1
                else:
                    self.__fp += 1
            for tgt_subtoken in filter(lambda x: x not in self._ignore_idxs, target):
                if tgt_subtoken not in prediction:
                    self.__fn += 1


class VarNamingModel(
    AbstractNeuralModel[VarNamingSample, TensorizedVarNamingSample, VarNamingGraphModel]
):
    def __init__(self, gnn_model: GraphNeuralNetworkModel, decoder_model: RNNDecoderModel):
        super().__init__()
        self.__gnn_model = gnn_model
        self.__decoder_model = decoder_model

    IDENTIFIER_REGEX: Final = re.compile("[a-zA-Z][a-zA-Z0-9]*")

    @property
    def vocabulary(self):
        return self.__decoder_model.vocabulary

    @classmethod
    def __add_subtoken_vocab_nodes(cls, graph: GraphData[str, str]) -> None:
        if "NextToken" not in graph.edges:
            return
        all_token_nodes = set(chain(*graph.edges["NextToken"]))

        subtoken_edges: List[Tuple[int, int]] = []
        reverse_subtoken_edges: List[Tuple[int, int]] = []
        subtoken_node_ids: Dict[str, int] = {}

        for token_node_idx in all_token_nodes:
            token_text = graph.node_information[token_node_idx]
            if not cls.IDENTIFIER_REGEX.match(token_text):
                continue
            for subtoken in split_identifier_into_parts(token_text):
                subtoken_node_idx = subtoken_node_ids.get(subtoken)
                if subtoken_node_idx is None:
                    subtoken_node_idx = len(graph.node_information)
                    graph.node_information.append(subtoken)
                    subtoken_node_ids[subtoken] = subtoken_node_idx

                subtoken_edges.append((subtoken_node_idx, token_node_idx))
                reverse_subtoken_edges.append((token_node_idx, subtoken_node_idx))

        graph.edges["SubtokenOf"] = subtoken_edges
        graph.edges["reverseSubtokenOf"] = reverse_subtoken_edges

    def update_metadata_from(self, datapoint: VarNamingSample) -> None:
        graph = datapoint["ContextGraph"]
        name = datapoint["name"]
        graph_data = GraphData(
            node_information=[
                graph["NodeLabels"][str(i)] for i in range(len(graph["NodeLabels"]))
            ],
            edges=graph["Edges"],
            reference_nodes={},  # This is not needed for metadata loading
        )
        self.__add_subtoken_vocab_nodes(graph_data)
        self.__gnn_model.update_metadata_from(graph_data)
        self.__decoder_model.update_metadata_from(name)

    def build_neural_module(self) -> VarNamingGraphModel:
        gnn = self.__gnn_model.build_neural_module()
        decoder = self.__decoder_model.build_neural_module()
        return VarNamingGraphModel(gnn, decoder)

    def tensorize(self, datapoint: VarNamingSample) -> Optional[TensorizedVarNamingSample]:
        graph = datapoint["ContextGraph"]
        name = datapoint["name"]
        var_node_idxs = [int(k) for k, v in graph["NodeLabels"].items() if v == VAR_TOKEN]

        graph_data = GraphData(
            node_information=[
                graph["NodeLabels"][str(i)]
                for i in range(len(graph["NodeLabels"]))
            ],
            edges=graph["Edges"],
            reference_nodes={
                "var_node_idxs": var_node_idxs
            },
        )

        if graph_data is None:
            return None

        self.__add_subtoken_vocab_nodes(graph_data)
        tensorized_graph_data = self.__gnn_model.tensorize(graph_data)
        if tensorized_graph_data is None:
            return None

        return TensorizedVarNamingSample(
            graph=tensorized_graph_data,
            target_idxs=self.__decoder_model.tensorize(name)
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "graph_data": self.__gnn_model.initialize_minibatch(),
            "target": self.__decoder_model.initialize_minibatch()
        }

    def extend_minibatch_with(
            self, tensorized_datapoint: TensorizedVarNamingSample, partial_minibatch: Dict[str, Any]
    ) -> bool:
        continue_adding = self.__gnn_model.extend_minibatch_with(
            tensorized_datapoint.graph, partial_minibatch["graph_data"]
        )
        self.__decoder_model.extend_minibatch_with(tensorized_datapoint.target_idxs, partial_minibatch["target"])
        return continue_adding

    def finalize_minibatch(
            self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        return {
            "graph_data": self.__gnn_model.finalize_minibatch(
                accumulated_minibatch_data["graph_data"], device=device
            ),
            "target": self.__decoder_model.finalize_minibatch(
                accumulated_minibatch_data["target"], device=device
            )
        }
