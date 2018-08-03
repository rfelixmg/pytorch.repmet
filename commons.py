'''
Taken from the tf-magnet-loss-master on github:
https://github.com/pumpikano/tf-magnet-loss

'''

import numpy as np
import torch

from utils import ensure_tensor

class Loss(object):
    """Sample minibatches for magnet loss."""
    def __init__(self, labels, k, m, d):

        self.unique_classes = np.unique(labels)
        self.num_classes = self.unique_classes.shape[0]
        self.labels = labels

        self.k = k
        self.m = m
        self.d = d

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

    def predict(self, rep_data):
        """
        Predict the clusters that rep_data belongs to based on the clusters current positions
        :param rep_data:
        :return:
        """
        # sc = self.centroids - np.expand_dims(rep_data, 1)
        # sc = sc * sc
        # sc = sc.sum(2)

        sc = self.euclidean_distance(ensure_tensor(self.centroids).float().cuda(),
                                     ensure_tensor(rep_data).float().cuda().unsqueeze(1)).detach().cpu().numpy()

        preds = np.argmin(sc, 1)  # calc closest clusters

        # at this point the preds are the cluster indexs, so need to get classes
        for p in range(len(preds)):
            preds[p] = self.cluster_classes[preds[p]]  # first change to cluster index
            preds[p] = self.unique_classes[preds[p]]  # then convert into the actual class label
        return preds

    def calc_accuracy(self, rep_data, y):
        """
        Calculate the accuracy of reps based on the current clusters
        :param rep_data:
        :param y:
        :return:
        """
        preds = self.predict(rep_data)
        correct = preds == y
        acc = correct.astype(float).mean()
        return acc

    def loss(self, r, classes, clusters, cluster_classes, n_clusters, alpha=1.0):
        raise NotImplementedError

    def minibatch_loss(self, r, classes, m, d, alpha=1.0):
        raise NotImplementedError

    @staticmethod
    def magnet_dist(means, reps):
        sample_costs = means - reps.unsqueeze(1)
        sample_costs = sample_costs * sample_costs
        sample_costs = sample_costs.sum(2)
        return sample_costs

    @staticmethod
    def euclidean_distance(x, y):
        return torch.sum((x - y) ** 2, dim=2)

