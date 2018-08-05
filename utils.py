'''
Modified from the tf-magnet-loss-master on github:
https://github.com/pumpikano/tf-magnet-loss

'''

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.utils import linear_assignment_
from sklearn.manifold import TSNE
from scipy.stats import itemfreq
from sklearn.cluster import KMeans
from itertools import chain

import torch
from torchvision import transforms

# Visualization + Plotting + Graphing
def moving_average(a, n=3):
    # Adapted from http://stackoverflow.com/questions/14313510/does-numpy-have-a-function-for-calculating-moving-average
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_smooth(history_dict, savepath=None):
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for k, v in history_dict.items():
        if k == 'loss':
            ax1.plot(v, 'c', alpha=0.5)
            ax1.plot(moving_average(v, 20), 'b')
        elif k == 'train acc':
            ax2.plot(v, 'lightsalmon', alpha=0.5)
            ax2.plot(moving_average(v, 20), 'r')
        elif k == 'test acc':
            ax2.plot(v, 'palegreen', alpha=0.5)
            ax2.plot(moving_average(v, 20), 'g')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.tick_params('y')
    ax2.set_ylabel('Acc')
    ax2.tick_params('y')

    fig.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


def plot_cluster_data(losses, classes, title="", savepath=None):
    plt.clf()
    y_pos = np.arange(len(classes))

    plt.figure(figsize=(int(len(classes)*0.25), 4))
    plt.bar(y_pos, losses, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    # plt.ylabel('Loss')
    plt.title(title)

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


def show_images(H):
    # make a square grid
    num = H.shape[0]
    rows = int(np.ceil(np.sqrt(float(num))))

    fig = plt.figure(1, [10, 10])
    grid = ImageGrid(fig, 111, nrows_ncols=[rows, rows])

    for i in range(num):
        grid[i].axis('off')
        grid[i].imshow(H[i], cmap='Greys')

    # Turn any unused axes off
    for j in range(i, len(grid)):
        grid[j].axis('off')


def plot_embedding(X, y, imgs=None, title=None, savepath=None):
    # Adapted from http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # Add image overlays
    if imgs is not None and hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(imgs[i], cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


def graph(vectors, labels, cluster_centers=None, cluster_classes=None, savepath=None):

    if cluster_centers is not None:
        vectors = np.vstack((vectors, cluster_centers))
        labels = np.hstack((labels, cluster_classes))
    else:
        cluster_classes = []

    if vectors.shape[1] > 2:
        try:
            tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
            vectors = tsne.fit_transform(vectors)
        except ValueError:
            print('Value Error')

    # plt.figure(figsize=(6, 5))
    plt.figure()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple']

    clss = list(set(labels))
    for i in range(len(labels)):
        if i < (len(vectors)-len(cluster_classes)):
            plt.scatter(vectors[i, 0],
                        vectors[i, 1],
                        s=10,
                        facecolors='none',
                        edgecolors=colors[clss.index(labels[i]) % len(colors)],
                        label=labels[i])
        else:
            plt.scatter(vectors[i, 0],
                        vectors[i, 1],
                        marker='x',
                        c=colors[clss.index(labels[i])%len(colors)],
                        label=labels[i])

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.clf()


def zip_chain(a, b):
    return list(chain(*zip(a, b)))


def plot_metric(*args, **kwargs):
    name = args[0]
    plot_data = []
    for i in range(1, len(args), 2):
        metrics = args[i]
        d = [m[name] for m in metrics]
        color = args[i + 1]
        plot_data.extend(zip_chain(d, color * len(d)))

    plt.plot(*plot_data)
    if kwargs['title']:
        plt.title(kwargs['title'])
    plt.show()


def disp_inputs(input_tensor, labels):
    if len(input_tensor.shape) > 3:
        for input in input_tensor[:]:
            img = transforms.ToPILImage()(input.cpu())
            plt.imshow(img)
            plt.show()
    else:
        img = input_tensor
        plt.imshow(img)
        plt.show()


def get_indexs(labels, n_classes, n_samples, class_ids=None):
    sample_indexs = []
    if not class_ids:
        class_ids = random.sample(set(labels), n_classes)

    for pc in class_ids:
        indexs = set(np.arange(len(labels))[labels == pc])
        added = 0
        for l in range(int(n_samples/len(indexs))):  # if we want more samples than we actually have available
            sample_indexs += indexs
            added += len(indexs)
        sample_indexs += random.sample(indexs, n_samples-added)

    return sample_indexs, class_ids


# Evaluation
def compute_rand_index(emb, labels):
    """
    https://en.wikipedia.org/wiki/Rand_index
    """
    n = len(emb)
    k = np.unique(labels).size

    m = KMeans(k)
    m.fit(emb)
    emb_labels = m.predict(emb)

    agreements = 0
    for i, j in zip(*np.triu_indices(n, 1)):
        emb_same = emb_labels[i] == emb_labels[j]
        gt_same = labels[i] == labels[j]

        if emb_same == gt_same:
            agreements += 1

    return float(agreements) / (n * (n - 1) / 2)


def unsupervised_clustering_accuracy(emb, labels):
    """
    Calcs acc for set of embeddings but redoes kmeans for the number of classes in labels rather than the learnt clusters
    :param emb:
    :param labels:
    :return:
    """

    k = np.unique(labels).size
    kmeans = KMeans(n_clusters=k, max_iter=35, n_init=15, n_jobs=-1).fit(emb)
    emb_labels = kmeans.labels_
    G = np.zeros((k, k))
    for i in range(k):
        lbl = labels[emb_labels == i]
        uc = itemfreq(lbl)
        for uu, cc in uc:
            G[i, uu] = -cc
    A = linear_assignment_.linear_assignment(G)
    acc = 0.0
    for (cluster, best) in A:
        acc -= G[cluster, best]
    return acc / float(len(labels))


# Type changing
def ensure_numpy(x):
    if type(x).__module__ == np.__name__:
        return x
    elif type(x).__module__ == torch.__name__:
        return x.detach().cpu().numpy()
    elif type(x).__module__ == 'torch.nn.parameter':
        return x.data.cpu().numpy()


def ensure_tensor(x):
    if type(x).__module__ == torch.__name__:
        return x
    elif type(x).__module__ == np.__name__:
        return torch.from_numpy(x)
    elif type(x).__module__ == 'torch.nn.parameter':
        return x


# Dataset labels formating
def get_labels(dataset, numpy=True):
    y = []
    for i in range(len(dataset)):
        y.append(dataset[i][1])
    if numpy:
        return np.asarray(y)
    else:
        return y


# Get dataset inputs
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


# Calculating Embeddings
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
            initial_reps.append(ensure_numpy(net(get_inputs(dataset, indexs_inner))))

        return np.vstack(initial_reps)
    else:
        return ensure_numpy(net(get_inputs(dataset, indexs)))
        # return net(get_inputs(dataset, indexs)*255).detach().cpu().numpy()  # MNIST only learns when is 0-255 not 0-1

def compute_all_reps(net, dataset, chunk_size):
    """Compute representations for entire set in chunks (sequential non-shuffled batches).

    Basically just forwards the inputs through the net to get the embedding vectors in 'chunks'
    """
    return compute_reps(net, dataset, list(range(len(dataset))), chunk_size=chunk_size)