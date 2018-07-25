from __future__ import print_function

from os.path import join

from utils import *
from magnet_ops import *
from magnet_tools import *

from data.stanford_dogs import StanDogs
from models.mnist import MNISTEncoder

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms


DATA_PATH = '/media/hayden/UStorage/DATASETS/IMAGE'
DATA_PATH = '/media/hayden/Storage21/DATASETS/IMAGE'

# Define magnet loss parameters
m = 12
d = 4
k = 3
alpha = 1.0
batch_size = m * d

chunk_size = 400
n_plot = 500



# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

assert torch.cuda.is_available(), 'Error: CUDA not found!'
# device = torch.device("cuda:0") # Uncomment this to run on GPU

train_dataset = StanDogs(root=DATA_PATH,
                         train=True,
                         cropped=False,
                         transform=transforms.ToTensor(),
                         download=True)

test_dataset = StanDogs(root=DATA_PATH,
                         train=False,
                         cropped=False,
                         transform=transforms.ToTensor(),
                         download=True)

# test_dataset = torchvision.datasets.Omniglot(root=DATA_PATH,
#                                           transform=transforms.ToTensor(),
#                                            download=True)

# train_dataset = torchvision.datasets.MNIST(root='data/mnist/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='data/mnist/',
#                                           train=False,
#                                           transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


def compute_reps(extract_fn, X, chunk_size):
    """Compute representations for input in chunks."""
    chunks = int(ceil(float(X.shape[0]) / chunk_size))
    reps = []
    for i in range(chunks):
        start = i * chunk_size
        stop = start + chunk_size
        chunk_reps = extract_fn(torch.from_numpy(X[start:stop])).detach().cpu().numpy()
        reps.append(chunk_reps)
    return np.vstack(reps)

# Define training data
X = train_dataset.train_data
y = train_dataset.train_labels  # has 60,000 samples compared to tf version 55,000 this is due to val set?

X = X.numpy().astype(np.float32)
y = y.numpy().astype(np.uint8)

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
    inputs = torch.from_numpy(X[batch_example_inds]).cuda()
    labels = torch.from_numpy(batch_class_inds).cuda()

    outputs = net(inputs)
    batch_loss, batch_example_losses = minibatch_magnet_loss(outputs, labels, m, d, alpha)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    batch_loss = float(batch_loss.detach().cpu().numpy())
    batch_example_losses = batch_example_losses.detach().cpu().numpy()

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

imgs = train_dataset.train_data.numpy()[:n_plot]
imgs = np.reshape(imgs, [n_plot, 28, 28])
plot_embedding(initial_reps[:n_plot], train_dataset.train_labels.numpy()[:n_plot])

imgs = train_dataset.train_data.numpy()[:n_plot]
imgs = np.reshape(imgs, [n_plot, 28, 28])
plot_embedding(final_reps[:n_plot], train_dataset.train_labels.numpy()[:n_plot])