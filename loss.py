'''
Modified from the tf-magnet-loss-master on github:
https://github.com/pumpikano/tf-magnet-loss

'''

import numpy as np
import torch
from utils import ensure_tensor, ensure_numpy
import torch.nn.functional as F
from sklearn.utils import linear_assignment_
from scipy.stats import itemfreq
from sklearn.cluster import KMeans


class Loss(object):
    """Sample minibatches for magnet loss."""
    def __init__(self, set_y, k, m, d, measure='euclidean', alpha=1.0):

        self.unique_y = np.sort(np.unique(set_y))
        self.num_classes = self.unique_y.shape[0]
        self.set_y = set_y

        self.k = k
        self.m = m
        self.d = d
        self.alpha = alpha

        if measure == 'euclidean':
            self.calculate_distance = self.euclidean_distance
        elif measure == 'cosine':
            self.calculate_distance = self.cosine_distance

        self.centroids = None
        self.assignments = np.zeros_like(set_y, int)
        self.cluster_assignments = {}
        self.cluster_classes = np.repeat(self.unique_y, k)
        self.example_losses = None
        self.cluster_losses = None
        self.has_loss = None
        self.avg_variance = None

    def update_clusters(self, rep_data, max_iter=20):
        raise NotImplementedError

    def update_losses(self, indexes, losses):
        """
        Given a list of examples indexes and corresponding losses
        store the new losses and update corresponding cluster losses.
        """
        # Lazily allocate structures for losses
        if self.example_losses is None:
            self.example_losses = np.zeros_like(self.set_y, float)
            self.cluster_losses = np.zeros([self.k * self.num_classes], float)
            self.has_loss = np.zeros_like(self.set_y, bool)

        # Update example losses
        indexes = np.array(indexes)  # these are sample indexs in the dataset, relating to the batch this is called on
        self.example_losses[indexes] = losses  # add loss for each of the examples in the batch
        self.has_loss[indexes] = losses  # set booleans

        # Find affected clusters and update the corresponding cluster losses
        cluster_indexs = np.unique(self.assignments[indexes])  # what clusters are these batch samples a part of?
        for cluster_index in cluster_indexs:  # for each cluster
            cluster_example_indexs = self.assignments == cluster_index  # what samples are in this cluster?
            cluster_example_losses = self.example_losses[cluster_example_indexs]  # for the samples in this cluster what are their losses

            # Take the average loss in the cluster of examples for which we have measured a loss
            if len(cluster_example_losses[self.has_loss[cluster_example_indexs]]) > 0:
                self.cluster_losses[cluster_index] = np.mean(cluster_example_losses[self.has_loss[cluster_example_indexs]])  # set the loss of this cluster of the mean if the losses exist
            else:
                self.cluster_losses[cluster_index] = 0

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

    def predict(self, x, method='simple'):
        """
        Predict the clusters that x belongs to based on the clusters current positions
        :param x:
        :return:
        """
        # Compute squared distance of each example to each cluster centroid (euclid without the root)
        sample_costs = self.calculate_distance(ensure_tensor(self.centroids).cuda().float(),
                                               ensure_tensor(x).cuda().float())

        if method == 'simple':
            # Hey why not just take the closest cluster from the training
            scores, cluster_indexs = torch.min(sample_costs, 1)

        elif method == 'repmet':
            # From RepNet paper
            # Is equivalent to magnet when variance is the same (just doesn't first calc as probability distribution)

            variance = 0.5  # having as 0.5 rather than a big self.avg_variance causes all costs to be 0 after the exp
            var_normalizer = -1 / (2 * variance)

            # apply exp and variance normalisation
            sample_costs = torch.exp(var_normalizer * sample_costs)

            sample_costs = sample_costs.view(-1, self.num_classes, self.k)
            numerator = torch.sum(sample_costs, dim=2)

            scores, cluster_indexs = torch.max(numerator, 1)
            cluster_indexs = cluster_indexs * self.k

        elif method == 'magnet':
            # From the magnet loss paper
            # Is equivalent to simple when k=1
            # otherwise makes decision based on sum of dists per class if such clusters are within the top L

            L = 128  # maybe let parse in as arg, not crucial
            if self.avg_variance:
                variance = self.avg_variance
            else:
                variance = 0.5
            var_normalizer = -1 / (2 * variance)

            if L < sample_costs.shape[1]:
                # generate mask for closest L clusters or all if # clusters is less than L
                sortd, _ = sample_costs.sort(1)
                maxs = sortd[:, L]
                maxs = maxs.unsqueeze(1)
                mask = torch.zeros_like(sample_costs)
                mask[sample_costs < maxs] = 1
            else:
                mask = torch.ones_like(sample_costs)  # ie basically no mask :P

            # apply exp and variance normalisation
            sample_costs = torch.exp(var_normalizer * sample_costs)

            # apply the mask after the exp but before the summings
            sample_costs = sample_costs * mask

            denominator = torch.sum(sample_costs, dim=1, keepdim=True)

            sample_costs = sample_costs.view(-1, self.num_classes, self.k)
            numerator = torch.sum(sample_costs, dim=2)
            denominator = denominator.expand(-1, numerator.shape[1])

            # Compute example losses and total loss
            epsilon = 1e-8
            probs = numerator / (denominator + epsilon) + epsilon

            scores, cluster_indexs = torch.max(probs, 1)
            cluster_indexs = cluster_indexs * self.k

        # Convert from cluster indexs to class labels
        predictions = self.cluster_classes[cluster_indexs]

        return predictions

    def calc_accuracy(self, x, y, method='repmet'):
        """
        Calculate the accuracy of reps based on the current clusters
        :param x:
        :param y:
        :param method:
        :return:
        """

        if method == 'unsupervised':
            # Tries finding a cluster for each class, and then assigns cluster labels to each cluster based on the max
            # samples of a particular class in that cluster
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
        else:
            predictions = self.predict(x, method=method)
            correct = predictions == y
            return correct.astype(float).mean()

    def loss(self, x, y):
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

    @staticmethod
    # Helper to compute boolean mask for distance comparisons
    def comparison_mask(a_labels, b_labels):
        return torch.eq(a_labels.unsqueeze(1),
                        b_labels.unsqueeze(0))
