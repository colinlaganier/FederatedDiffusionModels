import math
import numpy as np
import os
import torch
from PIL import Image
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
import numpy as np
import os
import torch
import torch.nn as nn
from collections import namedtuple
from torch.hub import get_dir, download_url_to_file
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.datasets import EMNIST
from torchvision.transforms import functional as TF
from model import load_model

def balanced_split(dataset, num_splits, client_id):
    """
    Splits training data into client datasets with balanced classes
    
    Args:
        dataset (torch.utils.data.Dataset): training data
        num_splits (int): number of client datasets to split into
    Returns:
        client_data (list): list of client datasets
    """
    samples_per_class = len(dataset) // num_splits
    remainder = len(dataset) % num_splits
    num_classes = 10
    class_counts = [0] * num_classes # number of samples per class
    subset_indices = [[] for _ in range(num_splits)] # indices of samples per subset
    for i, (data, target) in enumerate(dataset):
        # Add sample to subset if number of samples per class is less than samples_per_class
        if class_counts[target] < samples_per_class:
            subset_indices[i % num_splits].append(i)
            class_counts[target] += 1
        elif remainder > 0:
            subset_indices[i % num_splits].append(i)
            class_counts[target] += 1
            remainder -= 1

    return Subset(dataset, subset_indices[int(client_id)])


class VGGFeatureExtractor(nn.Module):
    WEIGHTS_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"

    def __init__(self):
        super().__init__()
        self.model = self._load_model()

    def _load_model(self):
        model_path = os.path.join(get_dir(), os.path.basename(self.WEIGHTS_URL))
        if not os.path.exists(model_path):
            download_url_to_file(self.WEIGHTS_URL, model_path)
        model = torch.jit.load(model_path).eval()
        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
        return model

    def forward(self, x):
        return self.model(x, return_features=True)


def compute_distance(row_features, col_features, row_batch_size, col_batch_size, device):
    dist = []
    for row_batch in row_features.split(row_batch_size, dim=0):
        dist_batch = []
        for col_batch in col_features.split(col_batch_size, dim=0):
            dist_batch.append(torch.cdist(
                row_batch.to(device).unsqueeze(0),
                col_batch.to(device).unsqueeze(0)
            ).squeeze(0).cpu())
        dist_batch = torch.cat(dist_batch, dim=1)
        dist.append(dist_batch)
    dist = torch.cat(dist, dim=0)
    return dist


def to_uint8(x):
    return (x * 127.5 + 128).clamp(0, 255).to(torch.uint8)

Manifold = namedtuple("Manifold", ["features", "kth"])


class ManifoldBuilder:
    def __init__(
            self,
            data=None,
            model=None,
            features=None,
            extr_batch_size=128,
            max_sample_size=50000,
            nhood_size=3,
            row_batch_size=10000,
            col_batch_size=10000,
            random_state=1234,
            num_workers=0,
            device=torch.device("cuda:0")  # set to cuda if available for the best performance
    ):
        if features is None:
            num_extr_batches = math.ceil(max_sample_size / extr_batch_size)
            if model is None:
                if hasattr(data, "__getitem__") and hasattr(data, "__len__"):  # map-style dataset
                    data_size = len(data)
                    if data_size > max_sample_size:
                        np.random.seed(random_state)
                        inds = torch.as_tensor(np.random.choice(data_size, size=max_sample_size, replace=False))
                        data = Subset(data, indices=inds)

                    def dataloader():
                        _dataloader = DataLoader(
                            data, batch_size=extr_batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False, pin_memory=True)
                        for x in _dataloader:
                            if isinstance(x, (list, tuple)):
                                yield x[0]
                            else:
                                yield x
                else:
                    assert isinstance(data, (np.ndarray, torch.Tensor, str))
                    if isinstance(data, str) and os.path.exists(data):
                        fmt = data.split(".")[-1]
                        if fmt == "npy":
                            data = np.load(data)
                        elif fmt == "pt":
                            data = torch.load(data)
                    data = torch.as_tensor(data)
                    assert data.dtype == torch.uint8
                    data_size = data.shape[0]
                    if data_size > max_sample_size:
                        np.random.seed(random_state)
                        inds = torch.as_tensor(np.random.choice(data_size, size=max_sample_size, replace=False))
                        data = data[inds]

                    def dataloader():
                        for i in range(num_extr_batches):
                            if i == num_extr_batches - 1:
                                yield data[i * extr_batch_size: max_sample_size]
                            else:
                                yield data[i * extr_batch_size: (i + 1) * extr_batch_size]
            else:
                def dataloader():
                    for i in range(num_extr_batches):
                        if i == num_extr_batches - 1:
                            yield to_uint8(model.sample_x(max_sample_size - extr_batch_size * i))
                        else:
                            yield to_uint8(model.sample_x(extr_batch_size))

            self.op_device = input_device = device
            if isinstance(device, list):
                self.extractor = nn.DataParallel(VGGFeatureExtractor().to(device[0]), device_ids=device)
                self.op_device = device[0]
                input_device = "cpu"
            else:
                self.extractor = VGGFeatureExtractor().to(device)

            features = []
            with torch.inference_mode():
                for x in tqdm(dataloader(), desc="Extracting features", total=num_extr_batches):
                    features.append(self.extractor(x.to(input_device)).cpu())
            features = torch.cat(features, dim=0)
        else:
            assert isinstance(features, torch.Tensor) and features.grad_fn is None
        features = features.to(torch.float16)  # half precision for faster distance computation?

        self.nhood_size = nhood_size
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.device = device

        self.features = features
        self.kth = self.compute_kth(features)

    def compute_distance(self, row_features, col_features):
        return compute_distance(
            row_features, col_features,
            row_batch_size=self.row_batch_size, col_batch_size=self.col_batch_size, device=self.op_device)

    def compute_kth(self, row_features: torch.Tensor, col_features: torch.Tensor = None):
        if col_features is None:
            col_features = row_features
        kth = []
        for row_batch in tqdm(row_features.split(self.row_batch_size, dim=0), desc="Computing k-th radii"):
            dist_batch = self.compute_distance(row_features=row_batch, col_features=col_features)
            kth.append(dist_batch.to(torch.float32).kthvalue(self.nhood_size + 1, dim=1).values.to(torch.float16))  # nhood_size + 1 to exclude itself
        kth = torch.cat(kth)
        return kth

    def save(self, fpath):
        save_dir = os.path.dirname(fpath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.manifold, fpath)

    @property
    def manifold(self):
        return Manifold(features=self.features, kth=self.kth)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def calc_pr(manifold_1: Manifold, manifold_2: Manifold, row_batch_size: int, col_batch_size: int, device):
    """
    Args:
        manifold_1: generated manifold namedtuple(support points, radii of k-th neighborhood (inclusive))
        manifold_2: ground truth manifold namedtuple(support points, radii of k-th neighborhood (inclusive))
        row_batch_size: literally
        col_batch_size: literally

    Returns:
        precision and recall
    """
    # ======= precision ======= #
    pred = []
    for probe_batch in tqdm(manifold_1.features.split(row_batch_size), desc="Calculating precision"):
        dist_batch = compute_distance(
            probe_batch, manifold_2.features,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, device=device)
        pred.append((dist_batch <= manifold_2.kth.unsqueeze(0)).any(dim=1))
    precision = torch.cat(pred).to(torch.float32).mean()

    # ======= recall ======= #
    pred.clear()
    for probe_batch in tqdm(manifold_2.features.split(row_batch_size), desc="Calculating recall"):
        dist_batch = compute_distance(
            probe_batch, manifold_1.features,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, device=device)
        pred.append((dist_batch <= manifold_1.kth.unsqueeze(0)).any(dim=1))
    recall = torch.cat(pred).to(torch.float32).mean()

    return precision, recall

if __name__ == "__main__":


    parser = ArgumentParser()

    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--eval-batch-size", default=128, type=int)
    parser.add_argument("--eval-total-size", default=10000, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--nhood-size", default=3, type=int)
    parser.add_argument("--row-batch-size", default=10000, type=int)
    parser.add_argument("--col-batch-size", default=10000, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--precomputed-dir", default="./precomputed", type=str)
    parser.add_argument("--metrics", nargs="+", default=["pr"], type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--sample-folder", default="", type=str)
    parser.add_argument("--num-gpus", default=1, type=int)

    args = parser.parse_args()

    # root = os.path.expanduser(args.root)
    dataset = args.dataset
    print(f"Dataset: {dataset}")

    folder_name = os.path.basename(args.sample_folder.rstrip(r"\/"))
    op_device = device = input_device = model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    precomputed_dir = args.precomputed_dir
    eval_batch_size = args.eval_batch_size
    eval_total_size = args.eval_total_size
    num_workers = args.num_workers
    row_batch_size = args.row_batch_size
    col_batch_size = args.col_batch_size
    nhood_size = args.nhood_size

    ##################################################################################
    # Sample Diffusion model 

    #labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    counter = [0] * 10
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    #testset = torchvision.datasets.ImageFolder(root="../../Testing/data/cinic-10/federated/5/train/client_0", transform=transforms.ToTensor())
    mean, std = [0.5], [0.5]
    transform = Compose([lambda img: TF.rotate(img, -90), lambda img: TF.hflip(img), Resize(32), ToTensor(), Normalize(mean, std)])
    testset = EMNIST(root='./data', train=False, download=True, transform=transform, split='digits')
    testset = balanced_split(testset, 4, 0)
    print("Testset size: {}".format(len(testset)))

    model = load_model(1)
    checkpoint = torch.load("../checkpoints/20230825-164926/model_100.pth")
    model.load_state_dict(checkpoint)
    model.to("cuda:0")
    total_samples = 10000
    num_samples = 10000
    num_channels = 1
    #for i in range(1):
    noise = torch.randn(num_samples, num_channels, 32, 32).to(device)
    fakes_classes = torch.arange(10, device=device).repeat_interleave(num_samples // 10, 0)
    fakes = sample(model, noise, 500, 1., fakes_classes)

    # for idx, fake in enumerate(fakes):
    #     cls = idx // (num_samples // 10)
    #     fake = TF.to_pil_image(fake.cpu().add(1).div(2).clamp(0, 1)).save("./data/synthetic/centralized/10K/{}/{}.png".format(labels[cls],counter[cls]))
    #     counter[cls] += 1


    # # Evaluate FID
    real_num = len(testset)
    subset = torch.utils.data.Subset(testset, random.sample(range(real_num), min(total_samples, real_num)))
    real_loader = torch.utils.data.DataLoader(subset, batch_size=100)
    fake_dataset = torch.utils.data.TensorDataset(fakes, fakes_classes)
    fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=100)
    print("Computing FID")
    fid = FIDScorer().calculate_fid(real_loader, fake_loader, device=device)
    print("FID: {}".format(fid))
 
    def eval_pr():
        decimal_places = math.ceil(math.log(eval_total_size, 10))
        str_fmt = f".{decimal_places}f"
        _ManifoldBuilder = partial(
            ManifoldBuilder, extr_batch_size=eval_batch_size, max_sample_size=eval_total_size,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, nhood_size=nhood_size,
            num_workers=num_workers, device=model_device)
        manifold_path = os.path.join(precomputed_dir, f"pr_manifold_{dataset}.pt")
        if not os.path.exists(manifold_path):
            manifold_builder = _ManifoldBuilder(data=subset)
            manifold_builder.save(manifold_path)
            true_manifold = deepcopy(manifold_builder.manifold)
            del manifold_builder
        else:
            true_manifold = torch.load(manifold_path)
        gen_manifold = deepcopy(_ManifoldBuilder(data=fake_dataset).manifold)

        precision, recall = calc_pr(
            gen_manifold, true_manifold,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, device=op_device)
        return f"{precision:{str_fmt}}/{recall:{str_fmt}}"

    def warning(msg):
        def print_warning():
            print(msg)
        return print_warning

    result_dict = {"folder_name": folder_name}
    with open(os.path.join(os.path.dirname(args.sample_folder.rstrip(r"\/")), "metrics.txt"), "a") as f:
        for metric in args.metrics:
            result = {"pr": eval_pr}.get(metric, warning("Unsupported metric passed! Ignore."))()
            print(f"{metric.upper()}: {result}")
            result_dict[metric] = result
        f.write(str(result_dict))




