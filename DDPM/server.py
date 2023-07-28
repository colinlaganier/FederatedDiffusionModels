import argparse
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision
import flwr as fl
import random
from FIDScorer import FIDScorer
from data_utils import load_data
from utils import test, eval_mode, sample
from model import load_model


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help=f"gRPC server address (default: 0.0.0.0:8080)",
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
        "--dataset-path", type=str, required=True, help="Path to dataset (no default)"
    )
    args = parser.parse_args()

    # Load evaluation data
    _, testset = load_data(args.dataset_path, 0)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
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
        "epochs": str(1),
        "batch_size": str(100),
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
        loss = test(model, testloader, device=DEVICE)
        real_num = len(testset)
        num_samples = 2500
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

        metrics = {"fid" : float(fid)}
        return loss, metrics

    return evaluate


if __name__ == "__main__":
    main()