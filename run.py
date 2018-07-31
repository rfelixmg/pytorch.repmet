from __future__ import print_function

import os

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import configs
from data.stanford_dogs import StanDogs
from data.utils import get_inputs
from models.utils import compute_all_reps, compute_reps
from magnet_ops import *
from magnet_tools import *
from models.configs import model_params
from models.mnist import MNISTEncoder
from models.standogs import SDEncoder
from utils import *

assert torch.cuda.is_available(), 'Error: CUDA not found!'

# DATASET = 'MNIST'
# MODEL_ID = '001'
DATASET = 'STANDOGS'
MODEL_ID = '002'

if DATASET == 'MNIST':

    dataset = torchvision.datasets.MNIST(root=os.path.join(configs.general.paths.imagesets, 'MNIST'),
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)

    test_dataset = torchvision.datasets.MNIST(root=os.path.join(configs.general.paths.imagesets, 'MNIST'),
                                              train=False,
                                              transform=transforms.ToTensor())

    net = MNISTEncoder(model_params.emb_dim[MODEL_ID])

elif DATASET == 'STANDOGS':

    input_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, ratio=(1, 1.3333)),
        transforms.ToTensor()])

    dataset = StanDogs(root=configs.general.paths.imagesets,
                       train=True,
                       cropped=False,
                       transform=input_transforms,
                       download=True)

    test_dataset = StanDogs(root=configs.general.paths.imagesets,
                            train=False,
                            cropped=False,
                            transform=input_transforms,
                            download=True)

    print("Training set stats:")
    dataset.stats()
    print("Testing set stats:")
    test_dataset.stats()

    net = SDEncoder(model_params.emb_dim[MODEL_ID])

plots_path = os.path.join(configs.general.paths.graphing, "%s" % (MODEL_ID))
os.makedirs(plots_path, exist_ok=True)

plots_ext = '.pdf'
plots_ext = '.png'

m = model_params.m[MODEL_ID]
d = model_params.d[MODEL_ID]
k = model_params.k[MODEL_ID]
alpha = model_params.alpha[MODEL_ID]


batch_size = m * d

chunk_size = 32#128
n_plot_samples = 15 # samples per class to plot
n_plot_classes = 5

n_epochs = 15
n_iterations = int(ceil(float(len(dataset)) / batch_size))
n_steps = n_iterations * n_epochs
cluster_refresh_interval = n_iterations

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
plot_sample_indexs = get_plot_indexs(y, n_plot_classes, n_plot_samples)
plot_test_sample_indexs = get_plot_indexs(test_labels, n_plot_classes, n_plot_samples)

# lets plot the initial embeddings
cluster_classes = batch_builder.cluster_classes

#use this to get indexs (indx to match cluster classes) for class ids (plot_classes) that we are plotting
for i in range(len(cluster_classes)):
    cluster_classes[i] = batch_builder.unique_classes[cluster_classes[i]]

# plot it
graph(initial_reps[plot_sample_indexs], y[plot_sample_indexs],
      cluster_centers=batch_builder.centroids,
      cluster_classes=batch_builder.cluster_classes,
      savepath="%s/emb-initial%s" % (plots_path, plots_ext))

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
        if not i % int(n_iterations*.05):
            # calc test acc
            test_reps = compute_all_reps(net, test_dataset, chunk_size)
            test_accb = unsupervised_clustering_accuracy(test_reps, test_labels)
            test_acc = test_accb # batch_builder.calc_accuracy(test_reps, test_labels)

            print("Iteration %03d/%03d: Tr. L: %0.3f :: Tr. A: %0.3f :::: Te. A: %0.3f :: Te. A2: %0.3f" % (i, n_iterations, batch_loss, batch_acc, test_acc, test_accb))

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
          savepath="%s/emb-e%d%s" % (plots_path, e+1, plots_ext))

    graph(compute_reps(net, test_dataset, plot_test_sample_indexs, chunk_size=chunk_size),
          test_labels[plot_test_sample_indexs],
          cluster_centers=batch_builder.centroids,
          cluster_classes=batch_builder.cluster_classes,
          savepath="%s/test_emb-e%d%s" % (plots_path, e+1, plots_ext))

    plot_smooth({'loss': batch_losses,
                 'train acc': batch_accs,
                 'test acc': test_accs},
                savepath="%s/loss%s" % (plots_path, plots_ext))

final_reps = compute_all_reps(net, dataset, chunk_size)

# Plot curves and graphs
plot_smooth({'loss':batch_losses,
             'train acc':batch_accs,
             'test acc':test_accs},
            savepath="%s/loss%s" % (plots_path, plots_ext))

graph(final_reps[plot_sample_indexs], y[plot_sample_indexs], savepath="%s/emb-final%s" % (plots_path, plots_ext))
