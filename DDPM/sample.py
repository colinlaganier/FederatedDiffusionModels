from utils import test, eval_mode, sample
from model import load_model
import torch
# from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from torchvision import transforms
import torchvision
import random
from FIDScorer import FIDScorer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
counter = [0] * 10
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.ImageFolder(root="../../Testing/data/cinic-10/federated/5/train/client_0", transform=transforms.ToTensor())

model = load_model()
model.load_state_dict(torch.load("checkpoint/model.pth"))
model.to("cuda:0")
total_samples = 100000
num_samples = 2500
for i in range(40):
    noise = torch.randn(num_samples, 3, 32, 32).to(device)
    fakes_classes = torch.arange(10, device=device).repeat_interleave(num_samples // 10, 0)
    fakes = sample(model, noise, 500, 1., fakes_classes)

    for idx, fake in enumerate(fakes):
        cls = idx // (num_samples // 10)
        fake = TF.to_pil_image(fake.cpu().add(1).div(2).clamp(0, 1)).save("synthetic/5K/{}/{}.png".format(labels[cls],counter[cls]))
        counter[cls] += 1


# # Evaluate FID
# real_num = len(testset)
# subset = torch.utils.data.Subset(testset, random.sample(range(real_num), min(total_samples, real_num)))
# real_loader = torch.utils.data.DataLoader(subset, batch_size=100)
# # fake_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fakes, fakes_classes), batch_size=100)
# # fid = FIDScorer().calculate_fid(real_loader, fake_loader, device=device)
# fake_saved = torchvision.datasets.ImageFolder(root="synthetic", transform=transforms.ToTensor())
# fake_saved = torch.utils.data.DataLoader(fake_saved, batch_size=100)
# fid_2 = FIDScorer().calculate_fid(real_loader, fake_saved, device=device)
# # print("FID: {}".format(fid))
# print("FID_2: {}".format(fid_2))