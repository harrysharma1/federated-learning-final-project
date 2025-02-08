import numpy as np   

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
torch.manual_seed(50)
# download and transform dataset to tensor
dst = datasets.CIFAR100("~/.torch", download=True)

tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])

tt = transforms.ToPILImage()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")


# Neural network and helper functions from DLG
class Helper():
    def __init__(self):
        pass
    def label_to_onehot(self, target, num_classes=100):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def cross_entropy_for_onehot(self, pred, target):
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

    def weights_init(self, m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    
    def check_similarity(self, ground_truth, reconstructed):
        ground_truth = np.array(ground_truth)
        reconstructed = np.array(reconstructed)

        ground_truth = (ground_truth * 255).astype(np.uint8)
        reconstructed = (reconstructed * 255).astype(np.uint8)

        mse = mean_squared_error(ground_truth, reconstructed)
        psnr = peak_signal_noise_ratio(ground_truth, reconstructed)
        ssim, _ = structural_similarity(ground_truth, reconstructed, full=True, win_size=3)

        return mse, psnr, ssim
            
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

helper = Helper()   
net = LeNet().to(device)
net.apply(helper.weights_init)
criterion = helper.cross_entropy_for_onehot

total_images = 2
results = []

for idx in range(total_images):
    gt_data = tp(dst[idx][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[idx][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = helper.label_to_onehot(gt_label, num_classes=100)
    
    print("GT label is %d." % gt_label.item(), "\nOnehot label is %d." % torch.argmax(gt_onehot_label, dim=-1).item())

    # Compute original gradient 
    out = net(gt_data)
    y = criterion(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    # Share the gradients with other clients
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # Generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    
    print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())
    
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    history = []
    
    history = []
    for iters in range(300):
        def closure():
            optimizer.zero_grad()

            pred = net(dummy_data) 
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            
            grad_diff = 0
            grad_count = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
                grad_count += gx.nelement()
            grad_diff.backward()
            
            return grad_diff
        
        optimizer.step(closure)
        if iters % 10 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))
    print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

    reconstructed_data = history[-1]
    mse, psnr, ssim = helper.check_similarity(tt(gt_data[0].cpu()), reconstructed_data)
    results.append((mse, psnr, ssim))    
# Plot similarity metrics
mse_values, psnr_values, ssim_values = zip(*results)


print(f"Mean Squared Error: {mse_values}")
print(f"Peak Signal-to-Noise Ratio: {psnr_values}")
print(f"SSIM: {ssim_values}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(mse_values, marker='o')
plt.title('MSE')
plt.xlabel('Image Index')
plt.ylabel('MSE')

plt.subplot(1, 3, 2)
plt.plot(psnr_values, marker='o')
plt.title('PSNR')
plt.xlabel('Image Index')
plt.ylabel('PSNR')

plt.subplot(1, 3, 3)
plt.plot(ssim_values, marker='o')
plt.title('SSIM')
plt.xlabel('Image Index')
plt.ylabel('SSIM')

plt.tight_layout()
plt.show()