import argparse
import timeit
import warnings
from typing import Dict

import torch
import torchvision
import torch.optim as optim

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

from data_utils import load_data
from utils import train
from DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer
from model import load_model
from Scheduler import GradualWarmupScheduler
from config import modelConfig

warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DiffusionClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        cid: str,
        device: int,
        trainset: torchvision.datasets.folder.ImageFolder,
        testset: torchvision.datasets.folder.ImageFolder,
    ) -> None:
        self.cid = cid
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = load_model(modelConfig).to(self.device)
        
        # Set data
        self.trainset = trainset
        self.testset = testset
        
        # Training settings
        self.epoch = 0
        self.grad_clip = modelConfig["grad_clip"]

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
        
        self.cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer= self.optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
        
        self.warmUpScheduler = GradualWarmupScheduler(optimizer=self.optimizer, multiplier=modelConfig["multiplier"],
            warm_epoch=modelConfig["epoch"] // 10, after_scheduler=self.cosineScheduler)
        
        self.trainer = GaussianDiffusionTrainer(
            self.model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(self.device)   

        # self.evaluater = GaussianDiffusionTrainer()     


    def get_parameters(self,  ins: GetParametersIns) -> GetParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: NDArrays = self.model.get_weights()
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

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
        
        train(self.model, self.trainer, self.optimizer, self.warmUpScheduler, self.grad_clip, trainloader, epochs, self.device)

        # Return the refined weights and the number of examples used for training
        weights_prime: NDArrays = self.model.get_weights()
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
        # self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        # testloader = torch.utils.data.DataLoader(
        #     self.testset, batch_size=100, shuffle=False
        # )
        # loss = (eval_mode(self.model_ema))(test(self.model_ema, testloader, device=DEVICE))
        loss = 0
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
        "--dataset-path", type=str, required=True, help="Path to dataset (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Device (default: 0)"
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load data
    trainset, testset = load_data(args.dataset_path, args.cid)

    # Start client
    client = DiffusionClient(args.cid, args.device, trainset, testset)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client)


if __name__ == "__main__":
    main()