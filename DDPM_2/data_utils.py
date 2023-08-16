from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor

def get_mean_std(dataset_id):
    """Get mean and std for normalization."""
    if (dataset_id == "cifar10"):
        return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    elif (dataset_id == "cifar100"):
        return [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif (dataset_id == "cinic10"):
        return [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]

def load_data(path, client_id):
    """Load training and test set."""
    mean, std = get_mean_std("cinic10")
    transform = Compose([ToTensor(), Normalize(mean, std)])
    trainset = ImageFolder(path + "/train/client_" + str(client_id), transform=transform)
    testset = ImageFolder(path + "/test", transform=transform)
    return trainset, testset