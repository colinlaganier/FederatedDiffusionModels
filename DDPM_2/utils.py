import torch
from tqdm import trange
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np

def train(model, trainer, optimizer, warmUpScheduler, grad_clip, trainloader, num_epoch, device):
    """Train the model on the training set."""
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    for epoch in range(num_epoch):
        with tqdm(trainloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(images, labels).sum() / b ** 2.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss: ": loss.item(),
                    "img shape: ": images.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        # Save checkpoints
        # torch.save(net_model.state_dict(), os.path.join(
        #     modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))        

# def eval(modelConfig: Dict, model):
#     device = torch.device(device)
#     model.eval()

#     with torch.no_grad():
#         # Generate fake labels
#         labels = torch.arange(1,11).repeat_interleave(10, 0)
        
#         sampler = GaussianDiffusionSampler(
#             model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
#         # Sampled from standard normal distribution
#         noisyImage = torch.randn(
#             size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
#         saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
#         # save_image(saveNoisy, os.path.join(
#             # modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
#         sampledImgs = sampler(noisyImage, labels)
#         sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
#         print(sampledImgs)
#         # save_image(sampledImgs, os.path.join(
#         #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])




#         loss = 0
#         real_num = len(testset)
#         num_samples = 1000
#         steps = 500
#         eta = 1.
        
#         # Generate fake images
#         noise = torch.randn([num_samples, 3, 32, 32], device=DEVICE)
#         fakes_classes = torch.arange(10, device=DEVICE).repeat_interleave(100, 0)
#         fakes = sample(model, noise, steps, eta, fakes_classes)
        
#         subset = torch.utils.data.Subset(testset, random.sample(range(real_num), min(num_samples, real_num)))
#         real_loader = torch.utils.data.DataLoader(subset, batch_size=100)
#         fake_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(fakes, fakes_classes), batch_size=100)
#         fid = FIDScorer().calculate_fid(real_loader, fake_loader, device=DEVICE)
#         logger.add_scalar("fid", fid, server_round)

#         metrics = {"fid" : float(fid)}
#         # return loss, metrics
    
#         with tqdm(testloader, dynamic_ncols=True) as tqdmDataLoader:
#             for images, labels in tqdmDataLoader:

