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

    def loss(self, x, y):  # y isn't used as set out of batch is just based on means of x as the y's
        """Compute magnet loss.

        Given a tensor of features `x`, the assigned class for each example,
        compute the magnet loss according to equation (4) in
        http://arxiv.org/pdf/1511.05939v2.pdf.

        Args:
            x: A batch of features.
            y: Class labels for each example.

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