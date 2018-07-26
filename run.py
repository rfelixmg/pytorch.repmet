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
from torchvision.models.resnet import resnet50
import torchvision.transforms as transforms


DATA_PATH = '/media/hayden/UStorage/DATASETS/IMAGE'
DATA_PATH = '/media/hayden/Storage21/DATASETS/IMAGE'

plots_path = '/media/hayden/Storage21/MODELS/repmet/mnist/plots/'

# Define magnet loss parameters
m = 8#12
d = 8#4
k = 3
alpha = 1.0
batch_size = m * d

chunk_size = 128
n_plot = 500

input_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, ratio=(1, 1.3333)),
    transforms.ToTensor()])

input_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, ratio=(1, 1.3333)),
    transforms.ToTensor()])

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

assert torch.cuda.is_available(), 'Error: CUDA not found!'
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# dataset = StanDogs(root=DATA_PATH,
#                          train=True,
#                          cropped=False,
#                          transform=input_transforms,
#                          download=True)

# dataset = StanDogs(root=DATA_PATH,
#                          train=False,
#                          cropped=False,
#                          transform=input_transforms,
#                          download=True)

# test_dataset = torchvision.datasets.Omniglot(root=DATA_PATH,
#                                           transform=transforms.ToTensor(),
#                                            download=True)

dataset = torchvision.datasets.MNIST(root='data/mnist/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data/mnist/',
                                          train=False,
                                          transform=transforms.ToTensor())
#
# # # Define training data
# X = dataset.train_data
# y = dataset.train_labels  # has 60,000 samples compared to tf version 55,000 this is due to val set?

# X = X.numpy().astype(np.float32)
# y = y.numpy().astype(np.uint8)

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


# def compute_reps(extract_fn, X, chunk_size):
#     """Compute representations for input in chunks.
#
#     Basically just forwards the inputs through the net to get the embedding vectors in 'chunks'
#     """
#     chunks = int(ceil(float(X.shape[0]) / chunk_size))
#     reps = []
#     for i in range(chunks):
#         start = i * chunk_size
#         stop = start + chunk_size
#         chunk_reps = extract_fn(torch.from_numpy(X[start:stop])).detach().cpu().numpy()
#         reps.append(chunk_reps)
#     return np.vstack(reps)

def compute_reps(net, dataset, indexs):
    """Compute representations/embeddings

    :param net: The net to foward through
    :param dataset: the dataset
    :param indexs: the indexs of the samples to pass
    :return: embeddings as a numpy array
    """
    inputs = None
    for index in indexs:
        if index == indexs[0]:
            inputs = torch.unsqueeze(dataset[index][0], 0)
        else:
            inputs = torch.cat((inputs, torch.unsqueeze(dataset[index][0], 0)), 0)
    # return net(inputs).detach().cpu().numpy()
    return net(inputs*255).detach().cpu().numpy()  # MNIST only learns when is 0-255 not 0-1

def compute_all_reps(net, dataset, chunk_size):
    """Compute representations for entire set in chunks (sequential non-shuffled batches).

    Basically just forwards the inputs through the net to get the embedding vectors in 'chunks'
    """
    initial_reps = []
    for s in range(0, len(dataset), chunk_size):
        indexs = list(range(s, min(s + chunk_size, len(dataset))))
        initial_reps.append(compute_reps(net, dataset, indexs))

    return np.vstack(initial_reps)


# Define model and training parameters
emb_dim = 2#1024 #2
n_epochs = 15
n_iterations = int(ceil(float(len(dataset)) / batch_size))
n_steps = n_iterations * n_epochs
cluster_refresh_interval = n_iterations


# Model
net = MNISTEncoder(emb_dim)
# net = resnet50(pretrained=False)

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# Use the GPU
net.cuda()
cudnn.benchmark = True

# Loss
# criterion =
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# Get initial embedding
initial_reps = compute_all_reps(net, dataset, chunk_size)

# initial_reps = compute_reps(net, X, chunk_size)

y = []
for i in range(len(dataset)):
    y.append(dataset[i][1])
y = np.asarray(y)

# Create batcher
batch_builder = ClusterBatchBuilder(y, k, m, d)
batch_builder.update_clusters(initial_reps)

batch_losses = []
for e in range(n_epochs):
    print("======= epoch %d =======" % (e+1))
    for i in range(n_iterations):

        # Sample batch and do forward-backward
        batch_example_inds, batch_class_inds = batch_builder.gen_batch()
        inputs = dataset.train_data[batch_example_inds].float().cuda()
        # inputs = dataset[batch_example_inds].cuda()
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
        if not i % 100:
            print(i, batch_loss)

    plot_embedding(compute_reps(net, dataset, list(range(400))), y[:n_plot], savepath="%semb-e%d.pdf"%(plots_path,e+1))

    print('Refreshing clusters')
    reps = compute_all_reps(net, dataset, chunk_size)
    batch_builder.update_clusters(reps)


final_reps = compute_all_reps(net, dataset, chunk_size)

# Plot curves and graphs
plot_smooth(batch_losses, savepath="%sloss.pdf"%(plots_path))
plot_embedding(initial_reps[:n_plot], y[:n_plot], savepath="%semb-initial.pdf"%(plots_path))
plot_embedding(final_reps[:n_plot], y[:n_plot], savepath="%semb-final.pdf"%(plots_path))