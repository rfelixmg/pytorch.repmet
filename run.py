from __future__ import print_function

from os.path import join
import random
from tqdm import tqdm

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
# plots_path = '/media/hayden/Storage21/MODELS/repmet/standogs/plots/'

# Define magnet loss parameters
m = 8#12#8 # num clusters
d = 8#4#8 # num examples
k = 3
alpha = 1.0
batch_size = m * d

chunk_size = 32#128
n_plot_samples = 15 # samples per class to plot
n_plot_classes = 10

# Define model and training parameters
emb_dim = 2
n_epochs = 15


# Model
net = MNISTEncoder(emb_dim)
# net = resnet50(pretrained=False, num_classes=emb_dim)


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
#
# test_dataset = StanDogs(root=DATA_PATH,
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

n_iterations = int(ceil(float(len(dataset)) / batch_size))
n_steps = n_iterations * n_epochs
cluster_refresh_interval = n_iterations

def get_inputs(dataset, indexs):
    """
    Gets the input data from a dataset
    :param dataset: The dataset
    :param indexs: List of the sample indexs
    :return: A tensor with the inputs stacked
    """
    inputs = None
    c = 0
    for index in indexs:
        if c == 0:
            inputs = torch.unsqueeze(dataset[index][0], 0)
        else:
            inputs = torch.cat((inputs, torch.unsqueeze(dataset[index][0], 0)), 0)
        c += 1
    return inputs

def compute_reps(net, dataset, indexs, chunk_size=None):
    """Compute representations/embeddings

    :param net: The net to foward through
    :param dataset: the dataset
    :param indexs: the indexs of the samples to pass
    :return: embeddings as a numpy array
    """
    if chunk_size:
        initial_reps = []
        for s in range(0, len(indexs), chunk_size):
            indexs_inner = list(indexs[s:min(s + chunk_size, len(indexs))])
            initial_reps.append(net(get_inputs(dataset, indexs_inner)).detach().cpu().numpy())

        return np.vstack(initial_reps)
    else:
        # return net(get_inputs(dataset, indexs)).detach().cpu().numpy()
        return net(inputs*255).detach().cpu().numpy()  # MNIST only learns when is 0-255 not 0-1

def compute_all_reps(net, dataset, chunk_size):
    """Compute representations for entire set in chunks (sequential non-shuffled batches).

    Basically just forwards the inputs through the net to get the embedding vectors in 'chunks'
    """
    return compute_reps(net, dataset, list(range(len(dataset))), chunk_size=chunk_size)


net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# Use the GPU
net.cuda()
cudnn.benchmark = True

# Loss
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# Get initial embedding
initial_reps = compute_all_reps(net, dataset, chunk_size)

# Train labels from dataset (TODO make into a dataset variable / func)
y = []
for i in range(len(dataset)):
    y.append(dataset[i][1])
y = np.asarray(y)

# Test labels from dataset (TODO make into a dataset variable / func)
test_labels = []
for i in range(len(test_dataset)):
    test_labels.append(test_dataset[i][1])
test_labels = np.asarray(test_labels)

# Create batcher object (this also stores the cluster centroids than we move occasionally)
batch_builder = ClusterBatchBuilder(y, k, m, d)
batch_builder.update_clusters(initial_reps)

# Randomly sample the classes then the samples from each class to plot
plot_sample_indexs = []
plot_classes = random.sample(set(y), n_plot_classes)
for pc in plot_classes:
    plot_sample_indexs += random.sample(set(np.arange(len(y))[y==pc]), n_plot_samples)

# Randomly sample the classes then the samples from each class to plot (test set)
plot_test_sample_indexs = []
plot_test_classes = random.sample(set(test_labels), n_plot_classes)
for pc in plot_test_classes:
    plot_test_sample_indexs += random.sample(set(np.arange(len(test_labels))[test_labels==pc]), n_plot_samples)

# lets plot the initial embeddings
cluster_classes = batch_builder.cluster_classes

#use this to get indexs (indx to match cluster classes) for class ids (plot_classes) that we are plotting
for i in range(len(cluster_classes)):
    cluster_classes[i] = batch_builder.unique_classes[cluster_classes[i]]

# plot it
graph(initial_reps[plot_sample_indexs], y[plot_sample_indexs],
      cluster_centers=batch_builder.centroids,
      cluster_classes=batch_builder.cluster_classes,
      savepath="%semb-initial.png" % plots_path)


# print("Training set stats:")
# dataset.stats()
# print("Testing set stats:")
# test_dataset.stats()

# Lets setup the training loop
batch_losses = []
batch_accs = []
test_accs = []
test_acc = 0
for e in range(n_epochs):
    print("======= epoch %d =======" % (e+1))
    # for i in tqdm(range(n_iterations)):
    for i in range(n_iterations):
        # Sample batch and do forward-backward
        batch_example_inds, batch_class_inds = batch_builder.gen_batch()

        # Get inputs and and labels from the dataset
        inputs = get_inputs(dataset, batch_example_inds).cuda()
        labels = torch.from_numpy(batch_class_inds).cuda()

        # Calc the outputs (embs) and then the loss + accs
        outputs = net(inputs)
        batch_loss, batch_example_losses, batch_acc = minibatch_magnet_loss(outputs, labels, m, d, alpha)

        # pass the gradient and update
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # just changing some types
        batch_loss = float(batch_loss.detach().cpu().numpy())
        batch_example_losses = batch_example_losses.detach().cpu().numpy()

        # Update loss index
        batch_builder.update_losses(batch_example_inds, batch_example_losses)

        # store the stats to graph at end
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)
        test_accs.append(test_acc)
        if not i % 50:
            # calc test acc
            test_reps = compute_all_reps(net, test_dataset, chunk_size)
            test_acc = batch_builder.calc_accuracy(test_reps, test_labels)

            print("Iteration %03d/%03d: Tr. L: %0.3f :: Tr. A: %0.3f :::: Te. A: %0.3f" % (i, n_iterations, batch_loss, batch_acc, test_acc))

    print('Refreshing clusters')
    reps = compute_all_reps(net, dataset, chunk_size)
    batch_builder.update_clusters(reps)

    cluster_classes = batch_builder.cluster_classes

    #use this to get indexs (indx to match cluster classes) for class ids (plot_classes) that we are plotting
    for i in range(len(cluster_classes)):
        cluster_classes[i] = batch_builder.unique_classes[cluster_classes[i]]

    graph(compute_reps(net, dataset, plot_sample_indexs, chunk_size=chunk_size),
          y[plot_sample_indexs],
          cluster_centers=batch_builder.centroids,
          cluster_classes=batch_builder.cluster_classes,
          savepath="%semb-e%d.png"%(plots_path, e+1))

    graph(compute_reps(net, test_dataset, plot_test_sample_indexs, chunk_size=chunk_size),
          test_labels[plot_test_sample_indexs],
          cluster_centers=batch_builder.centroids,
          cluster_classes=batch_builder.cluster_classes,
          savepath="%stest_emb-e%d.png"%(plots_path, e+1))

final_reps = compute_all_reps(net, dataset, chunk_size)

# Plot curves and graphs
plot_smooth({'loss':batch_losses,
             'train acc':batch_accs,
             'test acc':test_accs},
            savepath="%sloss.png"%(plots_path))

# plot_embedding(initial_reps[plot_sample_indexs], y[plot_sample_indexs], savepath="%semb-initial.pdf"%(plots_path))
# plot_embedding(final_reps[plot_sample_indexs], y[plot_sample_indexs], savepath="%semb-final.pdf"%(plots_path))
graph(final_reps[plot_sample_indexs], y[plot_sample_indexs], savepath="%semb-final.png"%(plots_path))