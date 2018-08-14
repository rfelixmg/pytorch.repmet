import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F

from loss import Loss
from utils import ensure_tensor

class MagnetLoss(Loss):
    """Sample minibatches for magnet loss."""
    def __init__(self, set_y, k, m, d, measure='euclidean', alpha=1.0):
        super().__init__(set_y, k, m, d, measure, alpha)

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
        sq_dists = ((self.centroids[seed_cluster_index] - self.centroids) ** 2).sum(axis=1)

        # Assure only clusters of different class from seed are chosen
        sq_dists[self.cluster_classes[seed_cluster_index] == self.cluster_classes] = np.inf

        # Get top m-1 (closest) impostor clusters and add seed cluster too
        cluster_indexs = np.argpartition(sq_dists, self.m-1)[:self.m-1]
        cluster_indexs = np.concatenate([[seed_cluster_index], cluster_indexs])

        # Sample d examples uniformly from m clusters
        batch_indexes = np.empty([self.m * self.d], int)
        for i, ci in enumerate(cluster_indexs):
            x = np.random.choice(self.cluster_assignments[ci], self.d, replace=True)  # if clusters have less than d samples we need to allow repick
            # x = np.random.choice(self.cluster_assignments[c], self.d, replace=False)
            start = i * self.d
            stop = start + self.d
            batch_indexes[start:stop] = x

        # Translate class indexes to index for classes within the batch
        # ie. [y1, y7, y56, y21, y89, y21,...] > [0, 1, 2, 3, 4, 3, ...]
        class_inds = self.cluster_classes[cluster_indexs]

        return batch_indexes, np.repeat(class_inds, self.d)

    def loss(self, x, y):  # y isn't used as set out of batch is just based on means of x as the y's
        """Compute magnet loss.

        Given a tensor of features `r`, the assigned class for each example,
        the assigned cluster for each example, the assigned class for each
        cluster, the total number of clusters, and separation hyperparameter,
        compute the magnet loss according to equation (4) in
        http://arxiv.org/pdf/1511.05939v2.pdf.

        Note that cluster and class indexes should be sequential startined at 0.

        Args:
            r: A batch of features.
            classes: Class labels for each example.
            m: The number of clusters in the batch.
            d: The number of examples in each cluster.
            alpha: The cluster separation gap hyperparameter.

        Returns:
            total_loss: The total magnet loss for the batch.
            losses: The loss for each example in the batch.
            acc: The predictive accuracy of the batch
        """

        batch_inds = torch.arange(self.m).unsqueeze(1).repeat(1, self.d).view(self.m * self.d).cuda()

        # Take cluster means within the batch (u_hat_m's (m of them))
        cluster_examples = x.view(self.m, int(float(x.shape[0]) / float(self.m)), x.shape[1])
        cluster_means = cluster_examples.mean(1)

        # Compute squared distance of each example to each cluster centroid (euclid without the root)
        sample_costs = self.calculate_distance(cluster_means, x)

        # Select distances of examples to their own centroid
        intra_cluster_mask = self.comparison_mask(batch_inds, torch.arange(self.m).cuda())
        diff_class_mask = ~intra_cluster_mask
        intra_cluster_costs = torch.sum(intra_cluster_mask.float() * sample_costs, dim=1)

        # Compute variance of intra-cluster distances
        N = x.shape[0]
        variance = intra_cluster_costs.sum() / float((N - 1))
        var_normalizer = -1 / (2 * variance)

        if not self.avg_variance:
            self.avg_variance = variance
        else:
            self.avg_variance = (self.avg_variance + variance) / 2

        # Compute numerator
        numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha)

        # Compute denominator
        denom_sample_costs = torch.exp(var_normalizer * sample_costs)
        denominator = torch.sum(diff_class_mask.float() * denom_sample_costs, dim=1)

        # Compute example losses and total loss
        epsilon = 1e-8
        losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))
        total_loss = losses.mean()

        _, pred_inds = sample_costs.min(1)
        acc = torch.eq(batch_inds, pred_inds.float()).float().mean()

        return total_loss, losses, acc

    # def calc_accuracy(self, x, y, stored_clusters=False):
    #     L = 128
    #
    #     variance = self.avg_variance
    #     var_normalizer = -1 / (2 * variance)
    #
    #     # Compute squared distance of each example to each cluster centroid (euclid without the root)
    #     sample_costs = self.calculate_distance(ensure_tensor(self.centroids).cuda().float(),
    #                                            ensure_tensor(x).cuda().float())
    #
    #     # apply exp and variance normalisation
    #     sample_costs = torch.exp(var_normalizer * sample_costs)
    #
    #     denominator = torch.sum(sample_costs, dim=1, keepdim=True)
    #
    #     sample_costs = sample_costs.view(-1, self.num_classes, self.k)
    #     numerator = torch.sum(sample_costs, dim=2)
    #     denominator = denominator.expand(-1, numerator.shape[1])
    #
    #     # Compute example losses and total loss
    #     epsilon = 1e-8
    #     probs = numerator / (denominator + epsilon) + epsilon
    #
    #     _, ind = torch.max(probs, 1)
    #     # _, ind = torch.max(numerator, 1)
    #
    #     pred = self.cluster_classes[ind*self.k] # get cluster class id, based on index, *k to make up index from summing over k
    #     acc = np.mean(np.equal(y, pred).astype(float))
    #
    #     return acc