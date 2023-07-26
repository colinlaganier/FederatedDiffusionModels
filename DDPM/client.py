import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from model import Diffusion
from copy import deepcopy
from torchvision.datasets import ImageFolder
from utils import ema_update, eval_loss, train_mode, eval_mode

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_mean_std(dataset_id):
    if (dataset_id == "cifar10"):
        return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    elif (dataset_id == "cifar100"):
        return [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif (dataset_id == "cinic10"):
        return [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]

def load_data(path, client_id):
    """Load CIFAR-10 (training and test set)."""
    mean, std = get_mean_std("cinic10")
    transform = Compose([ToTensor(), Normalize(mean, std)])
    trainset = ImageFolder(path + "/" + client_id + "/train", transform=transform)
    testset = ImageFolder(path + "/" + client_id + "/test", transform=transform)
    return trainset, testset
    # trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    # testset = CIFAR10("./data", train=False, download=True, transform=trf)
    # return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

model = Diffusion().to(DEVICE)
model_ema = deepcopy(model)
trainloader, testloader = load_data()
scaler = torch.cuda.amp.GradScaler()
seed = 0
epoch = 0
# Use a low discrepancy quasi-random sequence to sample uniformly distributed
# timesteps. This considerably reduces the between-batch variance of the loss.
rng = torch.quasirandom.SobolEngine(1, scramble=True)
ema_decay = 0.998
# The number of timesteps to use when sampling
steps = 500
# The amount of noise to add each timestep when sampling
eta = 1.

def train(model, model_ema, trainloader, epochs, device):
    """Train the model on the training set."""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    for _ in range(epochs):        
        epoch += 1
        for reals,  classes in tqdm(trainloader):
            optimizer.zero_grad()
            reals = reals.to(DEVICE)
            classes = classes.to(DEVICE)

            # Evaluate the loss
            loss = eval_loss(model, rng, reals, classes, DEVICE)

            # Do the optimizer step and EMA update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
            scaler.update()

@torch.no_grad()
@torch.random.fork_rng()
@eval_mode(model_ema)
def test(model_ema, testloader):
    """Validate the model on the test set."""
    torch.manual_seed(seed)
    rng = torch.quasirandom.SobolEngine(1, scramble=True)
    total_loss = 0
    count = 0
    for i, (reals, classes) in enumerate(tqdm(testloader)):
        reals = reals.to(DEVICE)
        classes = classes.to(DEVICE)

        loss = eval_loss(model_ema, rng, reals, classes, DEVICE)

        total_loss += loss.item() * len(reals)
        count += len(reals)
    loss = total_loss / count
    return loss


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model_ema.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model_ema = deepcopy(model)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, model_ema, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = test(model_ema, testloader)
        return loss, len(testloader.dataset)


# Start Flower client
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=FlowerClient(),
# )
train(model, model_ema, trainloader, epochs=1)

def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # Load model and data
    model = cifar.load_model()
    model.to(DEVICE)
    trainset, testset = cifar.load_data()

    # Start client
    client = CifarClient(args.cid, model, trainset, testset)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()