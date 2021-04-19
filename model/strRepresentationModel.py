import logging
from typing import Any, Counter, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import BpeVocabulary, CharTensorizer, Vocabulary
from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.neuralmodels.gnn.structs import AbstractNodeEmbedder
from typing_extensions import Final, Literal

from utils import INIT_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN

Vocabulary.get_unk = lambda _: UNK_TOKEN
Vocabulary.get_pad = lambda _: PAD_TOKEN


class TokenUnitEmbedder(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, dropout_rate: float):
        super().__init__()
        self.__embeddings = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_size
        )
        nn.init.xavier_uniform_(self.__embeddings.weight)  # TODO: Reconsider later?
        self.__dropout_layer = nn.Dropout(p=dropout_rate)

    @property
    def embedding_layer(self) -> nn.Embedding:
        return self.__embeddings

    def forward(self, token_idxs: torch.Tensor) -> torch.Tensor:
        return self.__dropout_layer(self.__embeddings(token_idxs)).unsqueeze(0)  # [1, B, D]


class SubtokenUnitEmbedder(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            dropout_rate: float
    ):
        super().__init__()
        self.__embeddings = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_size
        )
        nn.init.uniform_(self.__embeddings.weight)
        self.__dropout_layer = nn.Dropout(p=dropout_rate)

    @property
    def embedding_layer(self) -> nn.Embedding:
        return self.__embeddings

    def forward(self, token_idxs: torch.Tensor) -> torch.Tensor:
        """
        :param token_idxs: The subtoken ids in a [max_num_subtokens, B] matrix.
        :param lengths: A [B]-sized vector containing the lengths
        :return: a [max_num_subtokens, B, D] matrix of D-sized representations, max_num_subtokens per input example.
        """
        embedded = self.__embeddings(token_idxs)  # [max_num_subtokens, B, D]
        return self.__dropout_layer(embedded)


class StrRepresentationModel(
    AbstractNeuralModel[str, List[int], Union[TokenUnitEmbedder, SubtokenUnitEmbedder]],
    AbstractNodeEmbedder,
):
    """
    A model that accepts strings and returns a single representation (embedding) for each one of them.
    """

    def __init__(
            self,
            *,
            token_splitting: Literal["token", "subtoken", "bpe"],
            embedding_size: int = 128,
            dropout_rate: float = 0.2,
            # Vocabulary Options
            vocabulary_size: int = 20000,
            min_freq_threshold: int = 5,
            # BPE/Subtoken Options
            max_num_subtokens: Optional[int] = 6
    ):
        super().__init__()
        self._splitting_kind: Final = token_splitting
        self.embedding_size: Final = embedding_size
        self.dropout_rate: Final = dropout_rate
        self.__vocabulary: Union[Vocabulary, BpeVocabulary, CharTensorizer]
        if token_splitting in {"bpe", "subtoken"}:
            self.max_num_subtokens: Final = max_num_subtokens
        self.max_vocabulary_size: Final = vocabulary_size
        self.min_freq_threshold: Final = min_freq_threshold

    LOGGER: Final = logging.getLogger(__name__)

    def representation_size(self) -> int:
        return self.embedding_size

    @property
    def splitting_kind(self) -> Literal["token", "subtoken", "bpe"]:
        return self._splitting_kind

    # region Metadata Loading
    def initialize_metadata(self) -> None:
        self.__tok_counter = Counter[str]()
        self.__tok_counter[INIT_TOKEN] = self.min_freq_threshold + 1
        self.__tok_counter[EOS_TOKEN] = self.min_freq_threshold + 1

    def update_metadata_from(self, datapoint: str) -> None:
        if self.splitting_kind in {"token", "bpe"}:
            self.__tok_counter[datapoint] += 1
        elif self.splitting_kind == "subtoken":
            self.__tok_counter.update(split_identifier_into_parts(datapoint))
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}".')

    def finalize_metadata(self) -> None:
        if self.splitting_kind in {"token", "subtoken"}:
            self.__vocabulary = Vocabulary.create_vocabulary(
                self.__tok_counter,
                max_size=self.max_vocabulary_size,
                count_threshold=self.min_freq_threshold,
                add_pad=True
            )
        elif self.splitting_kind == "bpe":
            self.__vocabulary = BpeVocabulary(self.max_vocabulary_size,
                                              unk_token=UNK_TOKEN,
                                              pad_token=PAD_TOKEN,
                                              eos_token=EOS_TOKEN,
                                              bos_token=INIT_TOKEN)
            self.__vocabulary.create_vocabulary(self.__tok_counter)
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}"')

        del self.__tok_counter

    def build_neural_module(self) -> Union[TokenUnitEmbedder, SubtokenUnitEmbedder]:
        if self.splitting_kind == "token":
            vocabulary_size = len(self.vocabulary)
            embedding_size = self.embedding_size
            return TokenUnitEmbedder(vocabulary_size, embedding_size, self.dropout_rate)
        elif self.splitting_kind in {"bpe", "subtoken"}:
            vocabulary_size = len(self.vocabulary)
            embedding_size = self.embedding_size
            return SubtokenUnitEmbedder(
                vocabulary_size,
                embedding_size,
                self.dropout_rate
            )
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}"')

    @property
    def vocabulary(self) -> Union[Vocabulary, BpeVocabulary, CharTensorizer]:
        return self.__vocabulary

    # endregion

    # region Tensorization
    def tensorize(self, datapoint: str, return_str_rep: bool = False):
        if self.splitting_kind == "token":
            token_idxs = self.vocabulary.get_id_or_unk(datapoint)
            str_repr = datapoint
        elif self.splitting_kind == "subtoken":
            subtoks = split_identifier_into_parts(datapoint)
            if len(subtoks) == 0:
                subtoks = [Vocabulary.get_unk()]
            token_idxs = self.vocabulary.get_id_or_unk_multiple(subtoks)
        elif self.splitting_kind == "bpe":
            if len(datapoint) == 0:
                datapoint = "<empty>"
            token_idxs = self.vocabulary.get_id_or_unk_for_text(datapoint)
            if return_str_rep:  # Do _not_ compute for efficiency
                str_repr = self.vocabulary.tokenize(datapoint)
        else:
            raise ValueError(f'Unrecognized token splitting method "{self.splitting_kind}".')

        if return_str_rep:
            return token_idxs, str_repr
        return token_idxs

    # endregion

    # region Minibatching
    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"token_idxs": []}

    def extend_minibatch_with(
            self, tensorized_datapoint: Union[int, List[int]], partial_minibatch: Dict[str, Any]
    ) -> bool:
        partial_minibatch["token_idxs"].append(tensorized_datapoint)
        return True

    def finalize_minibatch(
            self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        bos_idx = self.vocabulary.token_to_id[INIT_TOKEN]
        eos_idx = self.vocabulary.token_to_id[EOS_TOKEN]
        if self.splitting_kind == "token":
            token_idxs = accumulated_minibatch_data["token_idxs"]
            return {
                "token_idxs": torch.tensor(
                    [[bos_idx] * len(token_idxs), token_idxs], dtype=torch.int64, device=device
                ),
            }
        elif self.splitting_kind in {"subtoken", "bpe"}:
            max_num_subtokens = max(len(t) for t in accumulated_minibatch_data["token_idxs"])
            if self.max_num_subtokens is not None:
                max_num_subtokens = min(max_num_subtokens, self.max_num_subtokens)

            subtoken_idxs = np.zeros(
                (max_num_subtokens, len(accumulated_minibatch_data["token_idxs"])), dtype=np.int32
            )
            lengths = np.empty(len(accumulated_minibatch_data["token_idxs"]), dtype=np.int32)
            for i, subtokens in enumerate(accumulated_minibatch_data["token_idxs"]):
                idxs = [bos_idx] + subtokens + [eos_idx]
                if len(idxs) > max_num_subtokens:
                    idxs = idxs[:max_num_subtokens]
                    idxs[-1] = eos_idx
                subtoken_idxs[: len(idxs), i] = idxs
                lengths[i] = len(idxs)

            return {
                "token_idxs": torch.tensor(subtoken_idxs, dtype=torch.int64, device=device),
                "lengths": torch.tensor(lengths, dtype=torch.int64, device=device),
            }
        else:
            raise Exception("Non-reachable state.")

    # endregion
