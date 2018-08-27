import torch
import torch.nn.functional as F
import torch.nn as nn

from loss import Loss
from utils import ensure_tensor, ensure_numpy


class MyLoss1(Loss):

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
            self.centroids = nn.Parameter(
                torch.zeros((self.num_classes * self.k, set_x.shape[1]), requires_grad=True).cuda())

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

        # Compute the mask selecting the sample_costs related to each class(e)=class(cluster/rep)
        intra_cluster_mask = self.comparison_mask(y, torch.from_numpy(self.cluster_classes).cuda())

        # make a mask for blocking out cluster / reps of classes that aren't represented in the batch
        intra_cluster_mask_b, _ = intra_cluster_mask.max(0)
        intra_cluster_mask_b = intra_cluster_mask_b.expand_as(distances)

        # Combine Masks into one
        inter_cluster_mask = (~intra_cluster_mask)*intra_cluster_mask_b

        # Get the furthest samples per rep with class(e)=class(rep)
        per_cluster_max, _ = (intra_cluster_mask.float() * distances).max(0)

        # Minus these max's from the sample costs and then minus alpha (-dists within cluster_max+alpha per cluster)
        sample_costs = distances - per_cluster_max.expand_as(distances) - self.alpha

        # hard code 0.5 [as suggested in paper] but seems to now work as well as the calculated variance in my exp
        variance = 0.5
        var_normalizer = -1 / (2 * variance**2)

        if not self.avg_variance:
            self.avg_variance = variance
        else:
            self.avg_variance = (self.avg_variance + variance) / 2

        # The var normaliser flips the -dists to dists before relu the costs so negatives go to 0 (outside cmax+alpha)
        sample_costs = F.relu(var_normalizer * sample_costs)

        # Sum across the clusters/reps (adds up for each sample the dists they are within cluster_max+alpha per cluster)
        losses = (inter_cluster_mask.float() * sample_costs).sum(1) / intra_cluster_mask_b.float().sum(1)

        total_loss = losses.mean()

        _, preds = distances.min(1)
        preds = ensure_tensor(self.cluster_classes[preds]).cuda()  # convert from cluster ids to class ids
        acc = torch.eq(y, preds).float().mean()

        return total_loss, losses, acc