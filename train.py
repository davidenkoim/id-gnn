import logging
import warnings
from os.path import join
from pathlib import Path

import hydra
import torch
import wandb
from ptgnn.baseneuralmodel import ModelTrainer
from ptgnn.baseneuralmodel.utils.data import LazyDataIterable
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import StrElementRepresentationModel
from ptgnn.neuralmodels.gnn import GraphNeuralNetworkModel
from ptgnn.neuralmodels.gnn.messagepassing import MlpMessagePassingLayer, GatedMessagePassingLayer, GruGlobalStateUpdate
from ptgnn.neuralmodels.gnn.messagepassing.residuallayers import ConcatResidualLayer, MeanResidualLayer
from ptgnn.neuralmodels.reduceops import WeightedSumVarSizedElementReduce
from torch import nn

from model.rnnDecoderModel import RNNDecoderModel
from model.strRepresentationModel import StrRepresentationModel
from model.varNamingModel import VarNamingModel
from utils import load_from_folder, log_run

warnings.filterwarnings("ignore")


@hydra.main(config_path='configs', config_name='config')
def train(cfg):
    if cfg.show_in_wandb:
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.WARNING)
        # Initializing wandb logger.
        name_of_run = cfg.model.name_of_run
        wandb.init(project="IdTransformer", config=cfg, group="GNN", name=name_of_run)

    training_data = LazyDataIterable(lambda: load_from_folder(join(cfg.dataset.path, "train"),
                                                              shuffle=True))
    validation_data = LazyDataIterable(lambda: load_from_folder(join(cfg.dataset.path, "validation"),
                                                                shuffle=False))
    test_data = LazyDataIterable(lambda: load_from_folder(join(cfg.dataset.path, "test"),
                                                          shuffle=False))

    model_path = Path(cfg.model.filename)
    assert model_path.name.endswith(".pkl.gz"), "model filename must have a `.pkl.gz` suffix."

    initialize_metadata = True
    restore_path = cfg.model.restore_path
    if restore_path and cfg.model.use_checkpoint:
        initialize_metadata = False
        model, nn = VarNamingModel.restore_model(Path(restore_path))
    else:
        nn = None
        model = create_var_naming_gnn_model(cfg.model)

    def create_optimizer(parameters):
        return torch.optim.Adam(parameters, lr=cfg.model.max_lr)

    trainer = ModelTrainer(
        model,
        model_path,
        max_num_epochs=int(cfg.model.max_epochs),
        minibatch_size=int(cfg.model.minibatch_size),
        optimizer_creator=create_optimizer,
        clip_gradient_norm=1,
        target_validation_metric="accuracy",
        target_validation_metric_higher_is_better=True,
    )
    if nn is not None:
        trainer.neural_module = nn

    if cfg.show_in_wandb:
        trainer.register_train_epoch_end_hook(
            lambda model, nn, epoch, metrics: log_run("train", model, epoch, metrics)
        )
        trainer.register_validation_epoch_end_hook(
            lambda model, nn, epoch, metrics: log_run("val", model, epoch, metrics)
        )

    trainer.train(
        training_data,
        validation_data,
        validate_on_start=True,
        show_progress_bar=True,
        initialize_metadata=initialize_metadata,
        parallelize=cfg.model.parallelize,
        use_multiprocessing=cfg.model.use_multiprocessing
    )


def create_var_naming_gnn_model(model_cfg):
    hidden_state_size = int(model_cfg.hidden_state_size)
    dropout = float(model_cfg.dropout)

    def create_mlp_mp_layers(num_edges: int):
        def mlp_mp_constructor():
            return MlpMessagePassingLayer(
                input_state_dimension=hidden_state_size,
                message_dimension=hidden_state_size,
                output_state_dimension=hidden_state_size,
                num_edge_types=num_edges,
                message_aggregation_function="max",
                dropout_rate=dropout,
            )

        def mlp_mp_after_res_constructor():
            return MlpMessagePassingLayer(
                input_state_dimension=2 * hidden_state_size,
                message_dimension=2 * hidden_state_size,
                output_state_dimension=hidden_state_size,
                num_edge_types=num_edges,
                message_aggregation_function="max",
                dropout_rate=dropout,
            )

        r1 = ConcatResidualLayer(hidden_state_size)
        r2 = ConcatResidualLayer(hidden_state_size)
        return [
            r1.pass_through_dummy_layer(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            r1,
            mlp_mp_after_res_constructor(),
            r2.pass_through_dummy_layer(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            mlp_mp_constructor(),
            r2,
            mlp_mp_after_res_constructor(),
        ]

    def create_ggnn_mp_layers(num_edges: int):
        ggnn_mp = GatedMessagePassingLayer(
            state_dimension=hidden_state_size,
            message_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="sum",
            dropout_rate=0.01,
        )
        r1 = MeanResidualLayer(hidden_state_size)
        r2 = MeanResidualLayer(hidden_state_size)

        def global_update():
            return GruGlobalStateUpdate(
                global_graph_representation_module=WeightedSumVarSizedElementReduce(hidden_state_size),
                input_state_size=hidden_state_size,
                summarized_state_size=hidden_state_size,
                dropout_rate=dropout,
            )

        return [
            r1.pass_through_dummy_layer(),
            r2.pass_through_dummy_layer(),
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            global_update(),
            ggnn_mp,
            r1,
            ggnn_mp,
            ggnn_mp,
            ggnn_mp,
            global_update(),
            ggnn_mp,
            r2,
        ]

    if model_cfg.mp_type == "mlp":
        create_mp_layers = create_mlp_mp_layers
    elif model_cfg.mp_type == "ggnn":
        create_mp_layers = create_ggnn_mp_layers
    else:
        ValueError('mp_type must be in ["mlp", "ggnn"]')
        return
    return VarNamingModel(
        gnn_model=GraphNeuralNetworkModel(
            node_representation_model=StrElementRepresentationModel(
                embedding_size=hidden_state_size, token_splitting="subtoken"
            ),
            message_passing_layer_creator=create_mp_layers,
            max_nodes_per_graph=int(model_cfg.max_nodes_per_graph),
            max_graph_edges=int(model_cfg.max_graph_edges),
            introduce_backwards_edges=False,
            add_self_edges=False,
            stop_extending_minibatch_after_num_nodes=int(model_cfg.stop_extending_minibatch_after_num_nodes)
        ),
        decoder_model=RNNDecoderModel(target_representation_model=StrRepresentationModel(
            embedding_size=hidden_state_size, token_splitting="subtoken"),
            create_rnn=lambda: nn.GRU(input_size=hidden_state_size, hidden_size=hidden_state_size, dropout=dropout)
        )
    )


if __name__ == "__main__":
    train()
