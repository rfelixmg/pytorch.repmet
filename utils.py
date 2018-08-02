'''
Taken from the tf-magnet-loss-master on github:
https://github.com/pumpikano/tf-magnet-loss

'''

# import tensorflow as tf
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


# Model building blocks

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='SAME')


# Visualization

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


def plot_cluster_loss(losses, classes, savepath=None):
    plt.clf()
    y_pos = np.arange(len(classes))

    plt.bar(y_pos, losses, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    plt.ylabel('Loss')
    plt.title('Cluster Losses')

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

