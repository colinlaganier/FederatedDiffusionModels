from utils import test, eval_mode, sample
from model import load_model
import torch
# from torchvision.utils import save_image
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.transforms import functional as TF
from torchvision import transforms
import torchvision
import random
from FIDScorer import FIDScorer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

counter = [0] * 10

# CINIC-10
# mean, std [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
# transform = Compose([ToTensor(), Normalize(mean, std)])
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
#testset = torchvision.datasets.ImageFolder(root="../../Testing/data/cinic-10/federated/5/train/client_0", transform=transforms.ToTensor())
# if test set divide by 9 -- 90,000 

# EMNIST
mean, std = [0.5], [0.5]
transform = Compose([lambda img: TF.rotate(img, -90), lambda img: TF.hflip(img), Resize(32), ToTensor(), Normalize(mean, std)])
testset = EMNIST(root='./data', train=False, download=True, transform=transform, split='digits')
testset = balanced_split(testset, 3, 0)

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

for idx, fake in enumerate(fakes):
    cls = idx // (num_samples // 10)
    fake = TF.to_pil_image(fake.cpu().add(1).div(2).clamp(0, 1)).save("./data/synthetic/centralized/10K/{}/{}.png".format(labels[cls],counter[cls]))
    counter[cls] += 1


# # Evaluate FID
real_num = len(testset)
subset = torch.utils.data.Subset(testset, random.sample(range(real_num), min(total_samples, real_num)))
real_loader = torch.utils.data.DataLoader(subset, batch_size=100)
fake_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fakes, fakes_classes), batch_size=100)
fid = FIDScorer().calculate_fid(real_loader, fake_loader, device=device)
fake_saved = torchvision.datasets.ImageFolder(root="./data/synthetic/centralized/10K", transform=transforms.ToTensor())
fake_saved = torch.utils.data.DataLoader(fake_saved, batch_size=100)
fid_2 = FIDScorer().calculate_fid(real_loader, fake_saved, device=device)
print("FID: {}".format(fid))
print("FID_2: {}".format(fid_2))
