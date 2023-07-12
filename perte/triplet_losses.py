import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from itertools import combinations


class TripletLoss(nn.Module):
    """
    Triplet loss
    """

    def __init__(
        self,
        alpha: float = 0.5,
        reduction: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.device = device
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        d_p = torch.norm(anchor - positive, dim=1)
        d_n = torch.norm(anchor - negative, dim=1)

        losses = torch.max(
            d_p - d_n + self.alpha, torch.FloatTensor([0]).to(self.device)
        )

        if self.reduction == "mean":
            return losses.mean(), d_p.mean(), d_n.mean()

        return losses.sum(), d_p.mean(), d_n.mean()


class TripletSelector:
    """
    Implementation should return indices of anchors,
    positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that
    take embeddings and targets and return indices of triplets
    """

    def __init__(
        self,
        triplet_selector: TripletSelector,
        margin: float = 0.5,
        reduction="mean",
        device=torch.device("cpu"),
    ):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.device = device
        self.reduction = reduction

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if triplets.size()[0] != 0:
            triplets = triplets.to(self.device)
            anchor = embeddings[triplets[:, 0]]
            positive = embeddings[triplets[:, 1]]
            negative = embeddings[triplets[:, 2]]
            anchor = F.normalize(anchor)
            positive = F.normalize(positive)
            negative = F.normalize(negative)
            d_p = (anchor - positive).pow(2).sum(1)
            d_n = (anchor - negative).pow(2).sum(1)
            losses = F.relu(d_p - d_n + self.margin)
            if self.reduction == "mean":
                return losses.mean(), d_p.mean(), d_n.mean()
            return losses.sum(), d_p.mean(), d_n.mean()
        else:
            losses = torch.tensor([self.margin], requires_grad=True)
            d_p = torch.tensor([0.], requires_grad=True)
            d_n = torch.tensor([0.], requires_grad=True)
            return losses, d_p, d_n

def pdist(vectors):
    distance_matrix = (
        -2 * vectors.mm(torch.t(vectors))
        + vectors.pow(2).sum(dim=1).view(1, -1)
        + vectors.pow(2).sum(dim=1).view(-1, 1)
    )
    return distance_matrix


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(
                combinations(label_indices, 2)
            )  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [
                [anchor_positive[0], anchor_positive[1], neg_ind]
                for anchor_positive in anchor_positives
                for neg_ind in negative_indices
            ]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    if loss_values[hard_negative] > 0:
        return hard_negative
    else:
        return None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    if len(hard_negatives) > 0:
        return np.random.choice(hard_negatives)
    else:
        return None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(
        np.logical_and(loss_values < margin, loss_values > 0)
    )[0]
    if len(semihard_negatives) > 0:
        return np.random.choice(semihard_negatives)
    else:
        return None


class BatchHardTripletSelector(TripletSelector):
    def __init__(self, margin: float = 0.5, device=torch.device("cpu")):
        super(BatchHardTripletSelector, self).__init__()

        self.device = device
        self.margin = margin

    def get_triplets(self, embeddings, labels):

        embeddings = embeddings.to(self.device)
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.to(self.device)

        labels = labels.cpu().data.numpy()
        triplets = []
        for anchor, label in enumerate(labels):
            label_mask = np.where(labels == label)[0]
            negative_indices = np.where(labels != label)[0]
            positive_indices = label_mask[label_mask != anchor]

            ap_distances = distance_matrix[
                torch.LongTensor(np.array([anchor])), torch.LongTensor(positive_indices)
            ]
            an_distances = distance_matrix[
                torch.LongTensor(np.array([anchor])), torch.LongTensor(negative_indices)
            ]

            ap_distances = ap_distances.data.cpu().numpy()
            an_distances = an_distances.data.cpu().numpy()

            # hard is small neg and large pos
            nidx = np.argmin(an_distances)
            pidx = np.argmax(ap_distances)

            positive = positive_indices[pidx]
            negative = negative_indices[nidx]

            if ap_distances[pidx] - an_distances[nidx] + self.margin > 0:
                triplets.append([anchor, positive, negative])

        triplets = np.array(triplets)
        return torch.LongTensor(triplets)


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample
    (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for
    a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(
        self, negative_selection_fn, margin: float = 0.5, device=torch.device("cpu")
    ):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.device = device
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        embeddings = embeddings.to(self.device)
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = labels == label
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(
                combinations(label_indices, 2)
            )  # All anchor-positive pairs
            anchor_positives.extend(list(combinations(label_indices[-1::-1], 2)))
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[
                anchor_positives[:, 0], anchor_positives[:, 1]
            ]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = (
                    ap_distance
                    - distance_matrix[
                        torch.LongTensor(np.array([anchor_positive[0]])),
                        torch.LongTensor(negative_indices),
                    ]
                    + self.margin
                )
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append(
                        [anchor_positive[0], anchor_positive[1], hard_negative]
                    )

        triplets = np.array(triplets)
        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin: float = 0.5, device=torch.device("cpu")):
    return FunctionNegativeTripletSelector(
        negative_selection_fn=hardest_negative, margin=margin, device=device
    )


def RandomNegativeTripletSelector(margin: float = 0.5, device=torch.device("cpu")):
    return FunctionNegativeTripletSelector(
        negative_selection_fn=random_hard_negative, margin=margin, device=device
    )


def SemihardNegativeTripletSelector(margin: float = 0.5, device=torch.device("cpu")):
    return FunctionNegativeTripletSelector(
        negative_selection_fn=lambda x: semihard_negative(x, margin),
        margin=margin,
        device=device,
    )