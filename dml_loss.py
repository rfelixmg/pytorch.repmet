import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans

from loss import Loss
from utils import ensure_tensor

class DMLLoss(Loss):

    def __init__(self, labels, k, m, d):
        super().__init__(labels, k, m, d)

    def update_clusters(self, rep_data, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """
        # Lazily allocate tensor for centroids
        if self.centroids is None:
            self.centroids = np.zeros([self.num_classes * self.k, rep_data.shape[1]])

        for c in range(self.num_classes):
            class_mask = self.labels == self.unique_classes[c]  # build true/false mask for classes to allow us to extract them
            class_examples = rep_data[class_mask]  # extract the embds for this class
            kmeans = KMeans(n_clusters=self.k, init='k-means++', n_init=1, max_iter=max_iter)
            kmeans.fit(class_examples)  # run kmeans putting k clusters per class

            # Save cluster centroids for finding impostor clusters
            start = self.get_cluster_ind(c, 0)
            stop = self.get_cluster_ind(c, self.k)
            self.centroids[start:stop] = torch.from_numpy(kmeans.cluster_centers_)

            # Update assignments with new global cluster indexes (ie each sample in dataset belongs to cluster id x)
            self.assignments[class_mask] = self.get_cluster_ind(c, kmeans.predict(class_examples))

        # make leaf variable after editing it then wrap in param
        # TODO see if it works without the param wrap
        self.centroids = nn.Parameter(torch.cuda.FloatTensor(self.centroids).requires_grad_())

        # Construct a map from cluster to example indexes for fast batch creation (the opposite of assignmnets)
        for cluster in range(self.k * self.num_classes):
            cluster_mask = self.assignments == cluster
            self.cluster_assignments[cluster] = np.flatnonzero(cluster_mask)

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
            seed_cluster = np.random.choice(self.num_classes * self.k, p=p)
        else:
            seed_cluster = np.random.choice(self.num_classes * self.k)

        # Get imposter clusters by ranking centroids by distance
        sq_dists = ((self.centroids[seed_cluster] - self.centroids) ** 2).sum(1).detach().cpu().numpy()

        # Assure only clusters of different class from seed are chosen
        # added the int condition as it was returning float and picking up seed_cluster
        sq_dists[int(self.get_class_ind(seed_cluster)) == self.cluster_classes] = np.inf

        # Get top m-1 (closest) impostor clusters and add seed cluster too
        clusters = np.argpartition(sq_dists, self.m - 1)[:self.m - 1]
        clusters = np.concatenate([[seed_cluster], clusters])

        # Sample d examples uniformly from m clusters
        batch_indexes = np.empty([self.m * self.d], int)
        for i, c in enumerate(clusters):
            x = np.random.choice(self.cluster_assignments[c], self.d, replace=True)  # if clusters have less than d samples we need to allow repick
            # x = np.random.choice(self.cluster_assignments[c], self.d, replace=False)
            start = i * self.d
            stop = start + self.d
            batch_indexes[start:stop] = x

        # Get classes for the clusters
        class_inds = self.cluster_classes[clusters]
        # batch_class_inds = []
        # inds_map = {}
        # class_count = 0
        # for c in class_inds:
        #     if c not in inds_map:
        #         inds_map[c] = class_count
        #         class_count += 1
        #     batch_class_inds.append(inds_map[c])


        # return batch_indexes, np.repeat(batch_class_inds, self.d)
        return batch_indexes, np.repeat(class_inds, self.d)

    def loss(self, r, classes, clusters, cluster_classes, n_clusters, alpha=1.0):
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
            clusters: Cluster labels for each example.
            cluster_classes: Class label for each cluster.
            n_clusters: Total number of clusters.
            alpha: The cluster separation gap hyperparameter.

        Returns:
            total_loss: The total magnet loss for the batch.
            losses: The loss for each example in the batch.
            acc: The predictive accuracy of the batch
        """
        # Helper to compute boolean mask for distance comparisons
        def comparison_mask(a_labels, b_labels):
            return torch.eq(a_labels.unsqueeze(1),
                            b_labels.unsqueeze(0))

        # Take cluster means within the batch
        # tf.dynamic_partition used clusters which is just an array we create in minibatch_magnet_loss, so can just reshape
        # cluster_examples = r.view(n_clusters, int(float(r.shape[0])/float(n_clusters)), r.shape[1])
        # cluster_means = cluster_examples.mean(1)

        # Compute squared distance of each example to each cluster centroid (euclid without the root)
        sample_costs = self.euclidean_distance(self.centroids, r.unsqueeze(1).cuda())

        # Compute the two masks selecting the sample_costs related to each class(r)=class(cluster/rep)
        intra_cluster_mask = comparison_mask(classes, torch.from_numpy(self.cluster_classes).cuda())
        intra_cluster_mask_op = ~intra_cluster_mask

        # Compute the costs, we only want to take the min of the closest r to it's corresponding cluster/rep
        # therefore we make all values for other clusters bigger so they are ignored..
        intra_cluster_costs_ignore = (intra_cluster_mask_op.float() * (sample_costs.max() * 1.5))
        intra_cluster_costs_desired = intra_cluster_mask.float() * sample_costs
        intra_cluster_costs_together = intra_cluster_costs_ignore + intra_cluster_costs_desired
        intra_cluster_costs, _ = intra_cluster_costs_together.min(1)

        # Compute variance of intra-cluster distances
        N = r.shape[0]
        variance = intra_cluster_costs.sum() / float((N - 1))
        var_normalizer = -1 / (2 * variance**2)

        # Compute numerator
        numerator = torch.exp(var_normalizer * intra_cluster_costs - alpha)

        # Compute denominator
        # diff_class_mask = tf.to_float(tf.logical_not(comparison_mask(classes, cluster_classes)))
        diff_class_mask = (intra_cluster_mask_op).float()
        denom_sample_costs = torch.exp(var_normalizer * sample_costs)
        denominator = (diff_class_mask.cuda() * denom_sample_costs).sum(1)

        # Compute example losses and total loss
        epsilon = 1e-8
        losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))
        total_loss = losses.mean()

        _, preds = sample_costs.min(1)
        preds = ensure_tensor(self.cluster_classes[preds]).cuda()  # convert from cluster ids to class ids
        acc = torch.eq(classes, preds).float().mean()

        return total_loss, losses, acc


    def minibatch_loss(self, r, classes, m, d, alpha=1.0):
        """Compute minibatch magnet loss.

        Given a batch of features `r` consisting of `m` clusters each with `d`
        consecutive examples, the corresponding class labels for each example
        and a cluster separation gap `alpha`, compute the total magnet loss and
        the per example losses. This is a thin wrapper around `magnet_loss`
        that assumes this particular batch structure to implement minibatch
        magnet loss according to equation (5) in
        http://arxiv.org/pdf/1511.05939v2.pdf. The correct stochastic approximation
        should also follow the cluster sampling procedure presented in section
        3.2 of the paper.

        Args:
            r: A batch of features.
            classes: Class labels for each example.
            m: The number of clusters in the batch.
            d: The number of examples in each cluster.
            alpha: The cluster separation gap hyperparameter.

        Returns:
            total_loss: The total magnet loss for the batch.
            losses: The loss for each example in the batch.
        """
        # clusters = np.repeat(np.arange(m, dtype=np.int32), d)
        clusters = torch.arange(m).unsqueeze(1).repeat(1, d).view(m*d)
        cluster_classes = classes.view(m, d)[:, 0]

        return self.loss(r, classes, clusters, cluster_classes, m, alpha)

