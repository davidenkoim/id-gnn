from typing import Any, List, Dict, Union, Optional, NamedTuple

import torch
from ptgnn.baseneuralmodel import AbstractNeuralModel
from torch import nn

from model.strRepresentationModel import TokenUnitEmbedder, SubtokenUnitEmbedder, StrRepresentationModel
from utils import INIT_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


class RNNOutput(NamedTuple):
    logits: torch.Tensor
    h: torch.Tensor


class RNNDecoder(nn.Module):
    def __init__(self,
                 target_embedder: Union[TokenUnitEmbedder, SubtokenUnitEmbedder],
                 rnn: nn.RNNBase,
                 bos_idx: int,
                 eos_idx: int,
                 pad_idx: int,
                 unk_idx: int
                 ):
        super().__init__()
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

        self._target_embedder = target_embedder
        self._rnn = rnn
        self._fc = nn.Linear(rnn.input_size, target_embedder.embedding_layer.num_embeddings)

    def forward(self, input: torch.Tensor, h: Optional[torch.Tensor] = None) -> RNNOutput:
        assert h is not None, "Specify hidden state h!"
        embeddings = self._target_embedder(input)  # [max_num_subtokens, B] -> [max_num_subtokens, B, D]
        emb, h = self._rnn(embeddings, h)
        return RNNOutput(self._fc(emb), h)  # [max_num_subtokens, B, D] -> [max_num_subtokens, B, vocab_size]


class RNNDecoderModel(AbstractNeuralModel[str, List[int], RNNDecoder]):
    def __init__(self,
                 target_representation_model: StrRepresentationModel,
                 create_rnn):
        super().__init__()
        self.__target_embedding_model = target_representation_model
        self.__create_rnn = create_rnn

    @property
    def vocabulary(self):
        return self.__target_embedding_model.vocabulary

    def update_metadata_from(self, datapoint: str) -> None:
        self.__target_embedding_model.update_metadata_from(datapoint)

    def build_neural_module(self) -> RNNDecoder:
        vocab = self.__target_embedding_model.vocabulary
        decoder = RNNDecoder(self.__target_embedding_model.build_neural_module(),
                             self.__create_rnn(),
                             unk_idx=vocab.token_to_id.get(UNK_TOKEN),
                             pad_idx=vocab.token_to_id.get(PAD_TOKEN),
                             bos_idx=vocab.token_to_id.get(INIT_TOKEN),
                             eos_idx=vocab.token_to_id.get(EOS_TOKEN))
        del self.__create_rnn
        return decoder

    def tensorize(self, datapoint: str) -> Optional[List[int]]:
        return self.__target_embedding_model.tensorize(datapoint)

    def initialize_minibatch(self) -> Dict[str, Any]:
        return self.__target_embedding_model.initialize_minibatch()

    def extend_minibatch_with(self, tensorized_datapoint: List[int],
                              partial_minibatch: Dict[str, Any]) -> bool:
        return self.__target_embedding_model.extend_minibatch_with(tensorized_datapoint, partial_minibatch)

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]) -> Dict[
        str, Any]:
        return self.__target_embedding_model.finalize_minibatch(accumulated_minibatch_data, device)
