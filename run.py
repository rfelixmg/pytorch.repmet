from __future__ import print_function

from utils import *
from magnet_ops import *
from magnet_tools import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data')



# Define magnet loss parameters
m = 8
d = 8
k = 3
alpha = 1.0
batch_size = m * d

chunk_size = 400
n_plot = 500



# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

assert torch.cuda.is_available(), 'Error: CUDA not found!'
# device = torch.device("cuda:0") # Uncomment this to run on GPU

train_dataset = torchvision.datasets.MNIST(root='data/mnist/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data/mnist/',
                                          train=False,
                                          transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


class MNISTEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(MNISTEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7 * 7 * 64, emb_dim)

    def forward(self, x, norm=False):
        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        if norm:
            return F.normalize(out)
        return out

def compute_reps(extract_fn, X, chunk_size):
    """Compute representations for input in chunks."""
    chunks = int(ceil(float(X.shape[0]) / chunk_size))
    reps = []
    for i in range(chunks):
        start = i * chunk_size
        stop = start + chunk_size
        chunk_reps = extract_fn(Variable(torch.from_numpy(X[start:stop]))).cpu().data.numpy()
        reps.append(chunk_reps)
    return np.vstack(reps)

# Define training data
X = train_dataset.train_data
y = train_dataset.train_labels  # has 60,000 samples compared to tf version 55,000 this is due to val set?

X = X.numpy().astype(np.float32)
y = y.numpy().astype(np.uint8)

# X = Variable(torch.from_numpy(X))

# Define model and training parameters
emb_dim = 2
n_epochs = 15
epoch_steps = int(ceil(float(X.shape[0]) / batch_size))
n_steps = epoch_steps * n_epochs
cluster_refresh_interval = epoch_steps



# Model
net = MNISTEncoder(emb_dim)

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# Use the GPU
net.cuda()
cudnn.benchmark = True

# Loss
# criterion =
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


# Get initial embedding
extract = net
initial_reps = compute_reps(extract, X, chunk_size)

# Create batcher
batch_builder = ClusterBatchBuilder(y, k, m, d)
batch_builder.update_clusters(initial_reps)

batch_losses = []
for i in range(n_steps):

    # Sample batch and do forward-backward
    batch_example_inds, batch_class_inds = batch_builder.gen_batch()
    inputs = Variable(torch.from_numpy(X[batch_example_inds])).cuda()
    labels = Variable(torch.from_numpy(batch_class_inds)).cuda()

    outputs = net(inputs)
    # outputs = Variable(torch.FloatTensor([[-1.9277,0.8476],
    #                                      [-6.2712,0.3196],
    #                                      [-0.7184,-3.6873],
    #                                      [-3.8063,8.0471],
    #                                      [2.7446,3.1479],
    #                                      [-5.3863,1.2954]]))
    batch_loss, batch_example_losses = minibatch_magnet_loss(outputs, labels, m, d, alpha)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    batch_loss = batch_loss.data.cpu().numpy()[0]
    batch_example_losses = batch_example_losses.data.cpu().numpy()

    # Update loss index
    batch_builder.update_losses(batch_example_inds, batch_example_losses)

    batch_losses.append(batch_loss)
    if not i % 200:
        print(i, batch_loss)

    if not i % cluster_refresh_interval:
        # imgs = train_dataset.train_data[:n_plot]
        # imgs = np.reshape(imgs, [n_plot, 28, 28])
        # plot_embedding(compute_reps(extract, X, 400)[:n_plot], train_dataset.train_labels[:n_plot])

        print('Refreshing clusters')
        reps = compute_reps(extract, X, chunk_size)
        batch_builder.update_clusters(reps)

final_reps = compute_reps(extract, X, chunk_size)

# Plot loss curve
plot_smooth(batch_losses)

imgs = train_dataset.train_data[:n_plot]
imgs = np.reshape(imgs, [n_plot, 28, 28])
plot_embedding(initial_reps[:n_plot], train_dataset.train_labels[:n_plot])

imgs = train_dataset.train_data[:n_plot]
imgs = np.reshape(imgs, [n_plot, 28, 28])
plot_embedding(final_reps[:n_plot], train_dataset.train_labels[:n_plot])