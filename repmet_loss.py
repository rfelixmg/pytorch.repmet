import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans

from loss import Loss
from utils import ensure_tensor, ensure_numpy

class RepMetLoss(Loss):

    def __init__(self, set_y, k, m, d, measure='euclidean', alpha=1.0):
        super().__init__(set_y, k, m, d, measure, alpha)

    def update_clusters(self, set_x, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """
        super().update_clusters(set_x, max_iter=max_iter)

        # make leaf variable after editing it then wrap in param
        self.centroids = nn.Parameter(torch.cuda.FloatTensor(self.centroids).requires_grad_())

    # def gen_batch(self):
    #     """
    #     Sample a batch by first sampling a seed cluster proportionally to
    #     the mean loss of the clusters, then finding nearest neighbor
    #     "impostor" clusters, then sampling d examples uniformly from each cluster.
    #
    #     The generated batch will consist of m clusters each with d consecutive
    #     examples.
    #     """
    #
    #     # Sample seed cluster proportionally to cluster losses if available
    #     if self.cluster_losses is not None:
    #         p = self.cluster_losses / np.sum(self.cluster_losses)
    #         seed_cluster_index = np.random.choice(self.num_classes * self.k, p=p)
    #     else:
    #         seed_cluster_index = np.random.choice(self.num_classes * self.k)
    #
    #     # Get imposter clusters by ranking centroids by distance
    #     sq_dists = ensure_numpy(((self.centroids[seed_cluster_index] - self.centroids) ** 2).sum(1))
    #
    #     # Assure only clusters of different class from seed are chosen
    #     # added the int condition as it was returning float and picking up seed_cluster
    #     sq_dists[int(self.get_class_ind(seed_cluster_index)) == self.cluster_classes] = np.inf
    #
    #     # Get top m-1 (closest) impostor clusters and add seed cluster too
    #     cluster_indexs = np.argpartition(sq_dists, self.m - 1)[:self.m - 1]
    #     cluster_indexs = np.concatenate([[seed_cluster_index], cluster_indexs])
    #
    #     # Sample d examples uniformly from m clusters
    #     batch_indexes = np.empty([self.m * self.d], int)
    #     for i, ci in enumerate(cluster_indexs):
    #         x = np.random.choice(self.cluster_assignments[ci], self.d, replace=True)  # if clusters have less than d samples we need to allow repick
    #         # x = np.random.choice(self.cluster_assignments[c], self.d, replace=False)
    #         start = i * self.d
    #         stop = start + self.d
    #         batch_indexes[start:stop] = x
    #
    #     # Get classes for the clusters
    #     class_inds = self.cluster_classes[cluster_indexs]
    #
    #     return batch_indexes, np.repeat(class_inds, self.d)

    def loss(self, x, y):
        """Compute repmet loss.

        Given a tensor of features `r`, the assigned class for each example,
        the assigned cluster for each example, the assigned class for each
        cluster, the total number of clusters, and separation hyperparameter,
        compute the magnet loss according to equation (4) in
        http://arxiv.org/pdf/1511.05939v2.pdf.

        Note that cluster and class indexes should be sequential startined at 0.

        Args:
            r: A batch of features.
            classes: Class labels for each example.
            clusters: Cluster labels for each example.
            cluster_classes: Class label for each cluster.
            n_clusters: Total number of clusters.
            alpha: The cluster separation gap hyperparameter.

        Returns:
            total_loss: The total magnet loss for the batch.
            losses: The loss for each example in the batch.
            acc: The predictive accuracy of the batch
        """

        # Compute distance of each example to each cluster centroid (euclid without the root)
        sample_costs = self.calculate_distance(self.centroids, x)

        # Compute the two masks selecting the sample_costs related to each class(r)=class(cluster/rep)
        intra_cluster_mask = self.comparison_mask(y, torch.from_numpy(self.cluster_classes).cuda())
        intra_cluster_mask_op = ~intra_cluster_mask

        # Calculate the minimum distance to rep of different class
        # therefore we make all values for correct clusters bigger so they are ignored..
        intra_cluster_costs_ignore_op = (intra_cluster_mask.float() * (sample_costs.max() * 1.5))
        intra_cluster_costs_desired_op = intra_cluster_mask_op.float() * sample_costs
        intra_cluster_costs_together_op = intra_cluster_costs_ignore_op + intra_cluster_costs_desired_op
        min_non_match, _ = intra_cluster_costs_together_op.min(1)

        # Calculate the minimum distance to rep of same class
        # therefore we make all values for other clusters bigger so they are ignored..
        intra_cluster_costs_ignore = (intra_cluster_mask_op.float() * (sample_costs.max() * 1.5))
        intra_cluster_costs_desired = intra_cluster_mask.float() * sample_costs
        intra_cluster_costs_together = intra_cluster_costs_ignore + intra_cluster_costs_desired
        min_match, _ = intra_cluster_costs_together.min(1)

        # Compute variance of intra-cluster distances
        # N = x.shape[0]
        # variance = min_match.sum() / float((N - 1))
        variance = 0.5  # hard code 0.5 [as suggested in paper] but seems to now work as well as the calculated variance in my exp
        var_normalizer = -1 / (2 * variance)

        # Compute numerator
        numerator = torch.exp(var_normalizer * min_match)

        # Compute denominator
        # diff_class_mask = tf.to_float(tf.logical_not(comparison_mask(classes, cluster_classes)))
        diff_class_mask = intra_cluster_mask_op.float()
        denom_sample_costs = torch.exp(var_normalizer * sample_costs)
        denominator = (diff_class_mask.cuda() * denom_sample_costs).sum(1)

        # Compute example losses and total loss
        epsilon = 1e-8
        losses_eq4 = F.relu(min_match - min_non_match + self.alpha)
        # total_loss_eq4 = losses_eq4.mean()

        # Compute example losses and total loss
        losses_eq5 = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon) + self.alpha)
        # total_loss_eq5 = losses_eq5.mean()

        losses = losses_eq5#losses_eq4 + losses_eq5
        total_loss = losses.mean()

        _, preds = sample_costs.min(1)
        preds = ensure_tensor(self.cluster_classes[preds]).cuda()  # convert from cluster ids to class ids
        acc = torch.eq(y, preds).float().mean()

        return total_loss, losses, acc



