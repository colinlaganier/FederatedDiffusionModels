import argparse
import timeit
import warnings

import torch
import torchvision

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    GetParametersRes,
    GetParametersIns,
    Status,
    Code,
)
from model import Diffusion
from copy import deepcopy
from torchvision.datasets import ImageFolder
from data_utils import load_data
from utils import ema_update, eval_mode, train_mode, train, test

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiffusionClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        model: Diffusion,
        trainset: torchvision.datasets.folder.ImageFolder,
        testset: torchvision.datasets.folder.ImageFolder,
    ) -> None:
        self.cid = cid
        self.model = model
        self.model_ema = deepcopy(model)
        self.trainset = trainset
        self.testset = testset
        self.scaler = torch.cuda.amp.GradScaler()
        # Training settings
        self.seed = 0
        self.epoch = 0
        # Use a low discrepancy quasi-random sequence to sample uniformly distributed
        # timesteps. This considerably reduces the between-batch variance of the loss.
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.ema_decay = 0.998
        # The number of timesteps to use when sampling
        self.steps = 500
        # The amount of noise to add each timestep when sampling
        self.eta = 1.


    def get_parameters(self,  ins: GetParametersIns) -> GetParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: NDArrays = self.model_ema.get_weights()
        parameters = fl.common.ndarrays_to_parameters(weights)
        return GetParametersRes(status=Status(code=Code.OK, message="Success"),
                                parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: NDArrays = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        # Set model parameters
        self.model.set_weights(weights)
        self.model_ema.set_weights(weights)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True
        )
        train(self.model, self.model_ema, trainloader, epochs, self.epoch, self.rng, self.scaler, self.ema_decay, DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: NDArrays = self.model_ema.get_weights()
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            metrics=metrics,
            status=Status(code=Code.OK, message="Success"),
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_ndarrays(ins.parameters)

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=False
        )
        # loss = (eval_mode(self.model_ema))(test(self.model_ema, testloader, device=DEVICE))
        loss = test(self.model_ema, testloader, device=DEVICE)
        metrics = {}
        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=loss, num_examples=len(self.testset), metrics=metrics, status=Status(code=Code.OK, message="Success")
        )

def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help=f"gRPC server address (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--dataset-path", type=str, help="Path to dataset (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "dataset", type=str, choices=["emnist, cinic10"], default="emnist"
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data   
    model = Diffusion().to(DEVICE)
    # model_ema = deepcopy(model)
    if args.dataset == "emnist":
        trainset, testset = load_data(args.dataset, args.cid)
    else:
        trainset, testset = load_data(args.dataset, args.cid, args.dataset_path)

    # Start client
    client = DiffusionClient(args.cid, model, trainset, testset)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client)


if __name__ == "__main__":
    main()