import torch
import torch.nn.functional as F

from loss import Loss


class MagnetLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, x, y):
        # y is used to specify the sum constraint where classes different,
        # for the case where k>1 and multiple imposter classes are from the same class
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

        cluster_examples_y = y.view(self.m, int(float(y.shape[0]) / float(self.m)))
        cluster_ys, _ = cluster_examples_y.max(1)

        # Compute squared distance of each example to each cluster centroid (euclid without the root)
        distances = self.calculate_distance(cluster_means, x)

        # Select distances of examples to their own centroid
        intra_cluster_mask = self.comparison_mask(batch_inds, torch.arange(self.m).cuda())
        diff_class_mask = ~self.comparison_mask(y, cluster_ys)
        intra_cluster_costs = torch.sum(intra_cluster_mask.float() * distances, dim=1)

        # Compute variance of intra-cluster distances
        N = x.shape[0]
        variance = intra_cluster_costs.sum() / float((N - 1))
        var_normalizer = -1 / (2 * variance**2)  # unsure whether this should be squared, is in eq(5) but not in eq(4)

        if not self.avg_variance:
            self.avg_variance = variance
        else:
            self.avg_variance = (self.avg_variance + variance) / 2

        # Compute numerator
        numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha)

        # Compute denominator
        denom_sample_costs = torch.exp(var_normalizer * distances)
        denominator = torch.sum(diff_class_mask.float() * denom_sample_costs, dim=1)

        # Compute example losses and total loss
        epsilon = 1e-8
        losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))
        total_loss = losses.mean()

        _, pred_inds = distances.min(1)
        acc = torch.eq(batch_inds, pred_inds.float()).float().mean()

        return total_loss, losses, acc
