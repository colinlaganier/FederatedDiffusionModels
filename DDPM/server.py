import argparse
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (Parameters, Scalar)

import os 
import time
import torch
import torchvision
import flwr as fl
import random
from collections import OrderedDict
from FIDScorer import FIDScorer
from data_utils import load_data
from utils import test, eval_mode, sample
from model import load_model
import numpy as np
from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Federated Averaging strategy with save model functionality."""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)


        # Save the model
        if aggregated_parameters is not None:
            model = load_model()
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            torch.save(model.state_dict(), checkpoint_path + f"/model.pth")
            # Save the model with round number
            if (server_round in [25, 50, 100, 150, 200, 250]):
                torch.save(model.state_dict(), checkpoint_path + f"/model_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

# class SaveProxStrategy(fl.server.strategy.FedProx):
    

def main() -> None:
    """Start server and train five rounds."""
    global num_epochs
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:6060",
        help=f"gRPC server address (default: 0.0.0.0:6060)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds of federated learning (default: 1)",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of available clients used for fit/evaluate (default: 1.0)",
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=2,
        help="Minimum number of clients used for fit/evaluate (default: 2)",
    )
    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--dataset-path", type=str, required=False, help="Path to dataset (no default)"
    )
    parser.add_argument(
        "--num-clients", type=int, required=True, help="Number of clients (no default)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs (default: 1)",
    )
    parser.add_argument(
        "dataset", type=str, choices=["emnist, cinic10"], required=False, default="emnist"
    )
    args = parser.parse_args()
    num_epochs = args.epochs

    # Load evaluation data
    _, testset = load_data(args.dataset_path, 0)

    # Create strategy
    strategy = SaveModelStrategy(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=get_evaluate_fn(testset),
        on_fit_config_fn=fit_config,
    )
    

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy)


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config: Dict[str, fl.common.Scalar] = {
        "epoch_global": str(server_round),
        "epochs": str(num_epochs),
        "batch_size": str(128),
    }
    return config


def get_evaluate_fn(
    testset: torchvision.datasets.folder.ImageFolder,
) -> Callable[[fl.common.NDArray], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, parameters: fl.common.NDArray, config) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = load_model()
        model.set_weights(parameters)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        # loss = test(model, testloader, device=DEVICE)
        loss = 0
        real_num = len(testset)
        num_samples = 10000
        steps = 500
        eta = 1.
        
        # Generate fake images
        noise = torch.randn([num_samples, 3, 32, 32], device=DEVICE)
        fakes_classes = torch.arange(10, device=DEVICE).repeat_interleave(250, 0)
        fakes = sample(model, noise, steps, eta, fakes_classes)
        
        subset = torch.utils.data.Subset(testset, random.sample(range(real_num), min(num_samples, real_num)))
        real_loader = torch.utils.data.DataLoader(subset, batch_size=100)
        fake_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fakes, fakes_classes), batch_size=100)
        fid = FIDScorer().calculate_fid(real_loader, fake_loader, device=DEVICE)
        logger.add_scalar("fid", fid, server_round)

        metrics = {"fid" : float(fid)}
        return loss, metrics

    return evaluate


if __name__ == "__main__":
    global checkpoint_path, logger
    # Create checkpoint directory
    checkpoint_path = "../checkpoints/" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"{checkpoint_path}", exist_ok=True)
    # Create tensorboard writer
    logger = SummaryWriter()
    
    main()