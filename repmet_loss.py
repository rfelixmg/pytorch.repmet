import torch
import torch.nn.functional as F
import torch.nn as nn

from loss import Loss
from utils import ensure_tensor, ensure_numpy


class RepMetLoss(Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_clusters(self, set_x, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """

        if self.centroids is None:
            # make leaf variable after editing it then wrap in param
            self.centroids = nn.Parameter(torch.zeros((self.num_classes * self.k, set_x.shape[1]), requires_grad=True).cuda())

        super().update_clusters(set_x, max_iter=max_iter)


    def loss(self, x, y):
        """Compute repmet loss.

        Given a tensor of features `x`, the assigned class for each example,
        compute the repmet loss according to equations (5) in
        https://arxiv.org/pdf/1806.04728.pdf.

        Args:
            x: A batch of features.
            y: Class labels for each example.

        Returns:
            total_loss: The total magnet loss for the batch.
            losses: The loss for each example in the batch.
            acc: The predictive accuracy of the batch
        """

        # Compute distance of each example to each cluster centroid (euclid without the root)
        distances = self.calculate_distance(self.centroids, x)

        # Compute the two masks selecting the sample_costs related to each class(r)=class(cluster/rep)
        intra_cluster_mask = self.comparison_mask(y, torch.from_numpy(self.cluster_classes).cuda())
        intra_cluster_mask_op = ~intra_cluster_mask

        # Calculate the minimum distance to rep of same class
        # therefore we make all values for other clusters bigger so they are ignored..
        intra_cluster_costs_ignore = (intra_cluster_mask_op.float() * (distances.max() * 1.5))
        intra_cluster_costs_desired = intra_cluster_mask.float() * distances
        intra_cluster_costs_together = intra_cluster_costs_ignore + intra_cluster_costs_desired
        min_match, _ = intra_cluster_costs_together.min(1)

        d = ensure_numpy(distances)
        t = ensure_numpy(intra_cluster_costs_ignore)
        tt = ensure_numpy(intra_cluster_costs_desired)
        ttt = ensure_numpy(intra_cluster_costs_together)
        tttt = ensure_numpy(min_match)
        # Compute variance of intra-cluster distances
        # N = x.shape[0]
        # variance = min_match.sum() / float((N - 1))
        variance = 0.5  # hard code 0.5 [as suggested in paper]
        var_normalizer = -1 / (2 * variance**2)

        if not self.avg_variance:
            self.avg_variance = variance
        else:
            self.avg_variance = (self.avg_variance + variance) / 2

        # Compute numerator
        numerator = torch.exp(var_normalizer * min_match)

        # Compute denominator
        diff_class_mask = intra_cluster_mask_op.float()
        denom_sample_costs = torch.exp(var_normalizer * distances)
        denominator = (diff_class_mask.cuda() * denom_sample_costs).sum(1)

        # Compute example losses and total loss
        epsilon = 1e-8

        # Compute example losses and total loss
        losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon) + self.alpha)

        total_loss = losses.mean()

        _, preds = distances.min(1)
        preds = ensure_tensor(self.cluster_classes[preds]).cuda()  # convert from cluster ids to class ids
        acc = torch.eq(y, preds).float().mean()

        return total_loss, losses, acc


# Don't only take min of class in numerator and use all in denominator
class RepMetLoss2(Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_clusters(self, set_x, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """

        if self.centroids is None:
            # make leaf variable after editing it then wrap in param
            self.centroids = nn.Parameter(torch.zeros((self.num_classes * self.k, set_x.shape[1]), requires_grad=True).cuda())

        super().update_clusters(set_x, max_iter=max_iter)


    def loss(self, x, y):
        """Compute repmet loss.

        Given a tensor of features `x`, the assigned class for each example,
        compute the repmet loss version 2 see readme of details.

        Args:
            x: A batch of features.
            y: Class labels for each example.

        Returns:
            total_loss: The total magnet loss for the batch.
            losses: The loss for each example in the batch.
            acc: The predictive accuracy of the batch
        """

        # Compute distance of each example to each cluster centroid (euclid without the root)
        distances = self.calculate_distance(self.centroids, x)

        # Compute the mask selecting the distances related to each class(r)=class(cluster/rep)
        intra_cluster_mask = self.comparison_mask(y, torch.from_numpy(self.cluster_classes).cuda())

        # Compute variance of intra-cluster distances
        # N = x.shape[0]
        # variance = min_match.sum() / float((N - 1))
        variance = 0.5  # hard code 0.5 [as suggested in paper]
        var_normalizer = -1 / (2 * variance**2)

        if not self.avg_variance:
            self.avg_variance = variance
        else:
            self.avg_variance = (self.avg_variance + variance) / 2

        # Compute numerator
        numerator_pre_mask = torch.exp(var_normalizer * distances)
        numerator = (intra_cluster_mask.float() * numerator_pre_mask).sum(1)

        # Compute denominator
        denominator = numerator_pre_mask.sum(1)

        # Compute example losses and total loss
        epsilon = 1e-8

        # Compute example losses and total loss
        losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon) + self.alpha)

        total_loss = losses.mean()

        _, preds = distances.min(1)
        preds = ensure_tensor(self.cluster_classes[preds]).cuda()  # convert from cluster ids to class ids
        acc = torch.eq(y, preds).float().mean()

        return total_loss, losses, acc


# Don't only take min of class in numerator, still exclude from denominator though
class RepMetLoss3(Loss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_clusters(self, set_x, max_iter=20):
        """
        Given an array of representations for the entire training set,
        recompute clusters and store example cluster assignments in a
        quickly sampleable form.
        """

        if self.centroids is None:
            # make leaf variable after editing it then wrap in param
            self.centroids = nn.Parameter(torch.zeros((self.num_classes * self.k, set_x.shape[1]), requires_grad=True).cuda())

        super().update_clusters(set_x, max_iter=max_iter)


    def loss(self, x, y):
        """Compute repmet loss.

        Given a tensor of features `x`, the assigned class for each example,
        compute the repmet loss version 2 see readme of details.

        Args:
            x: A batch of features.
            y: Class labels for each example.

        Returns:
            total_loss: The total magnet loss for the batch.
            losses: The loss for each example in the batch.
            acc: The predictive accuracy of the batch
        """

        # Compute distance of each example to each cluster centroid (euclid without the root)
        sample_costs = self.calculate_distance(self.centroids, x)

        # Compute the two masks selecting the sample_costs related to each class(r)=class(cluster/rep)
        intra_cluster_mask = self.comparison_mask(y, torch.from_numpy(self.cluster_classes).cuda())

        # Compute variance of intra-cluster distances
        # N = x.shape[0]
        # variance = min_match.sum() / float((N - 1))
        variance = 0.5  # hard code 0.5 [as suggested in paper] but seems to now work as well as the calculated variance in my exp
        var_normalizer = -1 / (2 * variance**2)

        if not self.avg_variance:
            self.avg_variance = variance
        else:
            self.avg_variance = (self.avg_variance + variance) / 2

        # Compute numerator
        sample_costs_e = torch.exp(var_normalizer * sample_costs - self.alpha)
        numerator = (intra_cluster_mask.float() * sample_costs_e).sum(1)

        # Compute denominator
        denominator = sample_costs_e.sum(1)
        # Compute example losses and total loss
        epsilon = 1e-8

        # Compute example losses and total loss
        losses = F.relu(-torch.log(numerator / (denominator - numerator + epsilon) + epsilon))

        total_loss = losses.mean()

        _, preds = sample_costs.min(1)
        preds = ensure_tensor(self.cluster_classes[preds]).cuda()  # convert from cluster ids to class ids
        acc = torch.eq(y, preds).float().mean()

        return total_loss, losses, acc
