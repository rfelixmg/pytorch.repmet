'''
Modified from the tf-magnet-loss-master on github:
https://github.com/pumpikano/tf-magnet-loss

'''

import numpy as np
import torch
from utils import ensure_tensor
import torch.nn.functional as F
from sklearn.utils import linear_assignment_
from scipy.stats import itemfreq
from sklearn.cluster import KMeans


class Loss(object):
    """Sample minibatches for magnet loss."""
    def __init__(self, labels, k, m, d, measure='euclidean'):

        self.unique_classes = np.unique(labels)
        self.num_classes = self.unique_classes.shape[0]
        self.labels = labels

        self.k = k
        self.m = m
        self.d = d

        if measure == 'euclidean':
            self.calculate_distance = self.euclidean_distance
        elif measure == 'cosine':
            self.calculate_distance = self.cosine_distance

        self.centroids = None
        self.assignments = np.zeros_like(labels, int)
        self.cluster_assignments = {}
        self.cluster_classes = np.repeat(range(self.num_classes), k)
        self.example_losses = None
        self.cluster_losses = None
        self.has_loss = None

    def update_clusters(self, rep_data, max_iter=20):
        raise NotImplementedError

    def update_losses(self, indexes, losses):
        """
        Given a list of examples indexes and corresponding losses
        store the new losses and update corresponding cluster losses.
        """
        # Lazily allocate structures for losses
        if self.example_losses is None:
            self.example_losses = np.zeros_like(self.labels, float)
            self.cluster_losses = np.zeros([self.k * self.num_classes], float)
            self.has_loss = np.zeros_like(self.labels, bool)

        # Update example losses
        indexes = np.array(indexes)  # these are sample indexs in the dataset, relating to the batch this is called on
        self.example_losses[indexes] = losses  # add loss for each of the examples in the batch
        self.has_loss[indexes] = losses  # set booleans

        # Find affected clusters and update the corresponding cluster losses
        clusters = np.unique(self.assignments[indexes])  # what clusters are these batch samples a part of?
        for cluster in clusters:  # for each cluster
            cluster_inds = self.assignments == cluster  # what samples are in this cluster?
            cluster_example_losses = self.example_losses[cluster_inds]  # for the samples in this cluster what are their losses

            # Take the average loss in the cluster of examples for which we have measured a loss
            if len(cluster_example_losses[self.has_loss[cluster_inds]]) > 0:
                self.cluster_losses[cluster] = np.mean(cluster_example_losses[self.has_loss[cluster_inds]])  # set the loss of this cluster of the mean if the losses exist
            else:
                self.cluster_losses[cluster] = 0

    def gen_batch(self):
        raise NotImplementedError

    def get_cluster_ind(self, c, i):
        """
        Given a class index and a cluster index within the class
        return the global cluster index
        """
        return c * self.k + i

    def get_class_ind(self, c):
        """
        Given a cluster index return the class index.
        """
        return c / self.k

    def predict(self, x):
        """
        Predict the clusters that rep_data belongs to based on the clusters current positions
        :param rep_data:
        :return:
        """
        sc = self.calculate_distance(ensure_tensor(self.centroids).float().cuda(),
                                     ensure_tensor(x).float().cuda()).detach().cpu().numpy()


        preds = np.argmin(sc, 1)  # calc closest clusters

        # at this point the preds are the cluster indexs, so need to get classes
        for p in range(len(preds)):
            preds[p] = self.cluster_classes[preds[p]]  # first change to cluster index
            preds[p] = self.unique_classes[preds[p]]  # then convert into the actual class label
        return preds

    def calc_accuracy(self, x, y, stored_clusters=False):
        """
        Calculate the accuracy of reps based on the current clusters
        :param rep_data:
        :param y:
        :return:
        """
        if stored_clusters:
            preds = self.predict(x)
            correct = preds == y
            return correct.astype(float).mean()
        else:
            k = np.unique(y).size
            kmeans = KMeans(n_clusters=k, max_iter=35, n_init=15, n_jobs=-1).fit(x)
            emb_labels = kmeans.labels_
            G = np.zeros((k, k))
            for i in range(k):
                lbl = y[emb_labels == i]
                uc = itemfreq(lbl)
                for uu, cc in uc:
                    G[i, uu] = -cc
            A = linear_assignment_.linear_assignment(G)
            acc = 0.0
            for (cluster, best) in A:
                acc -= G[cluster, best]
            return acc / float(len(y))

    def loss(self, r, classes, clusters, cluster_classes, n_clusters, alpha=1.0):
        raise NotImplementedError

    def minibatch_loss(self, r, classes, m, d, alpha=1.0):
        raise NotImplementedError

    # @staticmethod
    # def magnet_dist(means, reps):
    #     sample_costs = means - reps.unsqueeze(1)
    #     sample_costs = sample_costs * sample_costs
    #     sample_costs = sample_costs.sum(2)
    #     return sample_costs

    @staticmethod
    def euclidean_distance(x, y):
        return torch.sum((x - y.unsqueeze(1)) ** 2, dim=2)

    @staticmethod
    def cosine_distance(x, y):
        return F.cosine_similarity(x.transpose(0, 1).unsqueeze(0), y.unsqueeze(2))


def unsupervised_clustering_accuracy(emb, labels):
    """
    Calcs acc for set of embeddings but redoes kmeans for the number of classes in labels rather than the learnt clusters
    :param emb:
    :param labels:
    :return:
    """

