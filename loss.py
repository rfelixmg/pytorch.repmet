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
    """Parent loss class with functions useful for both magnet and repmet losses."""
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

    def update_clusters(self, set_x, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """
        # Lazily allocate array for centroids
        if self.centroids is None:
            self.centroids = np.zeros([self.num_classes * self.k, set_x.shape[1]])

        # make sure they are numpy, will convert for repmet where they stored as torch tensors
        self.centroids = ensure_numpy(self.centroids)

        for c in range(self.num_classes):
            class_mask = self.set_y == self.unique_y[c]  # build true/false mask for classes to allow us to extract them
            class_examples = set_x[class_mask]  # extract the embds for this class
            kmeans = KMeans(n_clusters=self.k, init='k-means++', n_init=1, max_iter=max_iter)
            kmeans.fit(class_examples)  # run kmeans putting k clusters per class

            # Save cluster centroids for finding impostor clusters
            start = self.get_cluster_ind(c, 0)
            stop = self.get_cluster_ind(c, self.k)
            self.centroids[start:stop] = kmeans.cluster_centers_

            # Update assignments with new global cluster indexes (ie each sample in dataset belongs to cluster id x)
            self.assignments[class_mask] = self.get_cluster_ind(c, kmeans.predict(class_examples))

        # Construct a map from cluster to example indexes for fast batch creation (the opposite of assignmnets)
        for cluster_index in range(self.k * self.num_classes):
            cluster_mask = self.assignments == cluster_index
            self.cluster_assignments[cluster_index] = np.flatnonzero(cluster_mask)

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
        """
        Sample a batch by first sampling a seed cluster proportionally to
        the mean loss of the clusters, then finding nearest neighbor
        "impostor" clusters, then sampling d examples uniformly from each cluster.

        The generated batch will consist of m clusters each with d consecutive
        examples.
        """

        # Sample seed cluster proportionally to cluster losses if available
        if self.cluster_losses is not None:
            p = self.cluster_losses / np.sum(self.cluster_losses)
            seed_cluster_index = np.random.choice(self.num_classes * self.k, p=p)
        else:
            seed_cluster_index = np.random.choice(self.num_classes * self.k)

        # Get imposter clusters by ranking centroids by distance
        sq_dists = self.calculate_distance(ensure_tensor(self.centroids[seed_cluster_index]), ensure_tensor(self.centroids))
        sq_dists = np.squeeze(ensure_numpy(sq_dists))

        # Assure only clusters of different class from seed are chosen
        sq_dists[self.cluster_classes[seed_cluster_index] == self.cluster_classes] = np.inf

        # Get top m-1 (closest) impostor clusters and add seed cluster too
        cluster_indexs = np.argpartition(sq_dists, self.m - 1)[:self.m - 1]
        cluster_indexs = np.concatenate([[seed_cluster_index], cluster_indexs])

        # Sample d examples uniformly from m clusters
        batch_indexes = np.empty([self.m * self.d], int)
        for i, ci in enumerate(cluster_indexs):
            x = np.random.choice(self.cluster_assignments[ci], self.d,
                                 replace=True)  # if clusters have less than d samples we need to allow repick
            # x = np.random.choice(self.cluster_assignments[c], self.d, replace=False)
            start = i * self.d
            stop = start + self.d
            batch_indexes[start:stop] = x

        # Translate class indexes to index for classes within the batch
        # ie. [y1, y7, y56, y21, y89, y21,...] > [0, 1, 2, 3, 4, 3, ...]
        class_inds = self.cluster_classes[cluster_indexs]

        return batch_indexes, np.repeat(class_inds, self.d)

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
