from __future__ import print_function
from __future__ import division
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
torch.backends.cudnn.benchmark = True

import os
if not os.path.isdir("result_gaussian"):
    os.makedirs("result_gaussian")

# ---------------------------------------------------------------------------------------
thres = 1e-4  # loss threshold to stop
plot_result = True  # plot results?
device = torch.device("cuda")
max_iter= 100
batch_size = 200
nz = 2  # latent dimension
n = 2000  # number of samples in target domain
n_hidden = 512
d = 2  # data dimension

r = np.random.uniform(low=0.8, high=0.99, size=n)  # radius
theta = np.random.uniform(low=0, high=2 * np.pi, size=n)  # angle
x = np.sqrt(r) * np.cos(theta)
y = np.sqrt(r) * np.sin(theta)
y_t = np.concatenate((x[:, None], y[:, None]), axis=1)

y_t = torch.from_numpy(y_t).float().to(device) / 2

G = nn.Sequential(
    nn.Linear(nz, n_hidden),
    nn.ReLU(True),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(True),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(True),
    nn.Linear(n_hidden, d),
    nn.Tanh()
).to(device)


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.zero_()


memory_size = 100
early_end = (200, 320)

initialize_weights(G)

opt_g = torch.optim.Adam(G.parameters(), lr=1e-4)

psi = torch.zeros(len(y_t), requires_grad=True, device=device)
opt_psi = torch.optim.Adam([psi], lr=1e-1)

z_output = torch.randn(batch_size, nz, device=device)

print("Building C++ extension for W1 (requires PyTorch >= 1.0.0)...")
from torch.utils.cpp_extension import load
my_ops = load(name="my_ops", sources=["W1_extension/my_ops.cpp", "W1_extension/my_ops_kernel.cu"], verbose=False)
import my_ops
print("Building complete")


for it in range(max_iter):
    z_batch = torch.randn(batch_size, nz, device=device)

    # OTS
    ot_loss = []
    w1_estimate = []
    memory_p = 0
    memory_z = torch.zeros(memory_size, batch_size, nz)
    memory_y = torch.zeros(memory_size, batch_size, dtype=torch.long)
    for ots_iter in range(1, 2001):
        opt_psi.zero_grad()

        z_batch = torch.randn(batch_size, nz, device=device)
        y_fake = G(z_batch)

        score = -my_ops.l1_t(y_fake, y_t) - psi

        phi, hit = torch.max(score, 1)

        loss = torch.mean(phi) + torch.mean(psi)
        loss_primal = torch.mean(torch.abs(y_fake - y_t[hit])) * d

        loss_back = -torch.mean(psi[hit])  # equivalent to loss
        loss_back.backward()

        opt_psi.step()

        ot_loss.append(loss.item())
        w1_estimate.append(loss_primal.item())

        memory_z[memory_p] = z_batch
        memory_y[memory_p] = hit
        memory_p = (memory_p + 1) % memory_size

    # FIT
    g_loss = []
    for fit_iter in range(memory_size):
        opt_g.zero_grad()

        z_batch = memory_z[fit_iter].to(device)
        y_fake = G(z_batch)  # G_t(z)
        y0_hit = y_t[memory_y[fit_iter].to(device)]  # T(G_{t-1}(z))

        loss_g = torch.mean(torch.abs(y0_hit - y_fake)) * d
        loss_g.backward()
        opt_g.step()
        g_loss.append(loss_g.item())
    print(it, "FIT loss:", np.mean(g_loss))


    if plot_result:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
        plt.close('all')
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.title('before backprop')
        plt.xlim(xmin*5, xmax*5)
        plt.ylim(ymin*5, ymax*5)
        plt.scatter(z_batch[:, 0].detach().cpu(), z_batch[:, 1].detach().cpu(), marker='+', color=[0.05, 0.28, 0.63])

        y_plot = G(z_output)
        plt.subplot(122)
        plt.title('loss: {0:g}'.format(loss.item()))
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.scatter(y_t[:, 0].cpu(), y_t[:, 1].cpu(), marker='x', color=[0.8, 0.22, 0])
        plt.scatter(y_plot[:, 0].detach().cpu(), y_plot[:, 1].detach().cpu(), marker='+', color=[0.05, 0.28, 0.63])

        plt.savefig("./result_gaussian/iter_{}.jpg".format(it), format='jpg')
        pass
pass
