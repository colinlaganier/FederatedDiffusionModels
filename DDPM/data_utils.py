from torchvision.datasets import ImageFolder, EMNIST
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF

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

def dirichlet_split(dataset, num_splits, client_id, beta=0.1):
    """
    Splits training data into client datasets based Dirichlet distribution

    Args:
        dataset (torch.utils.data.Dataset): training data
        num_splits (int): number of client datasets to split into
        beta (float): concentration parameter of Dirichlet distribution
    Returns:
        client_data (list): list of client datasets       
    """
    # set seed for reproducibility
    np.random.seed(42)

    label_distributions = []
    # Generate label distributions for each class using Dirichlet distribution
    for y in range(len(dataset.classes)):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, num_splits)))

    labels = np.array(dataset.targets).astype(int)
    client_idx_map = {i: {} for i in range(num_splits)}
    client_size_map = {i: {} for i in range(num_splits)}

    for y in range(len(dataset.classes)):
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)

        # Sample number of samples for each client from label distribution
        sample_size = (label_distributions[y] * label_y_size).astype(int)
        sample_size[num_splits - 1] += len(label_y_idx) - np.sum(sample_size)
        for i in range(num_splits):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(num_splits):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i - 1] if i > 0 else 0):sample_interval[i]]


    client_i_idx = np.concatenate(list(client_idx_map[int(client_id)].values()))
    np.random.shuffle(client_i_idx)
    return Subset(dataset, client_i_idx)

def get_mean_std(dataset_id):
    """Get mean and std for normalization."""
    if (dataset_id == "cifar10"):
        return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    elif (dataset_id == "cifar100"):
        return [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif (dataset_id == "cinic10"):
        return [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
    elif (dataset_id == "emnist"):
        return (0.1307,), (0.3081,)

def load_data(dataset, client_id, path=None):
    """Load training and test set."""
    mean, std = get_mean_std(dataset)
    
    if dataset == "cifar10":
        transform = Compose([ToTensor(), Normalize(mean, std)])
        trainset = ImageFolder(path + "/train/client_" + str(client_id), transform=transform)
        testset = ImageFolder(path + "/test", transform=transform)
    else: 
        transform = Compose([lambda img: TF.rotate(img, -90),
                                lambda img: TF.hflip(img),
                                Resize(32), ToTensor(), Normalize(mean, std)])
        if client_id:               
            trainset = EMNIST(root='./data', train=True, download=True, transform=transform, split='digits')
            trainset = balanced_split(trainset, 5, client_id)
        else: 
            trainset = None

        testset = EMNIST(root='./data', train=False, download=True, transform=transform, split='mnist')
    
    return trainset, testset