# import tensorflow as tf
import torch
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np


def magnet_loss(r, classes, clusters, cluster_classes, n_clusters, alpha=1.0):
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
    cluster_examples = r.view(n_clusters, int(float(r.shape[0])/float(n_clusters)), r.shape[1])
    cluster_means = cluster_examples.mean(1)

    # Compute squared distance of each example to each cluster centroid (euclid without the root)
    sample_costs = euclidean_distance(cluster_means, r.unsqueeze(1))

    # Select distances of examples to their own centroid
    intra_cluster_mask = comparison_mask(clusters, torch.arange(n_clusters)).float()
    intra_cluster_costs = (intra_cluster_mask.cuda() * sample_costs).sum(1)

    # Compute variance of intra-cluster distances
    N = r.shape[0]
    variance = intra_cluster_costs.sum() / float((N - 1))
    var_normalizer = -1 / (2 * variance**2)

    # Compute numerator
    numerator = torch.exp(var_normalizer * intra_cluster_costs - alpha)

    # Compute denominator
    # diff_class_mask = tf.to_float(tf.logical_not(comparison_mask(classes, cluster_classes)))
    diff_class_mask = (~comparison_mask(classes, cluster_classes)).float()
    denom_sample_costs = torch.exp(var_normalizer * sample_costs)
    denominator = (diff_class_mask.cuda() * denom_sample_costs).sum(1)

    # Compute example losses and total loss
    epsilon = 1e-8
    losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))
    total_loss = losses.mean()

    _, preds = sample_costs.min(1)
    acc = torch.eq(classes, preds).float().mean()

    return total_loss, losses, acc


def minibatch_magnet_loss(r, classes, m, d, alpha=1.0):
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
    
    return magnet_loss(r, classes, clusters, cluster_classes, m, alpha)


def magnet_dist(means, reps):

    sample_costs = means - reps.unsqueeze(1)
    sample_costs = sample_costs * sample_costs
    sample_costs = sample_costs.sum(2)
    return sample_costs


def euclidean_distance(x, y):
    return torch.sum((x - y)**2, dim=2)