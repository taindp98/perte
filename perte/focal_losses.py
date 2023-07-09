import torch.nn.functional as F
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 0.0,
        eps: float = 1e-7,
        class_weights=None,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss


class FocalCosineLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1,
        gamma: float = 2,
        xent: float = 0.1,
        reduction: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.xent = xent

        self.y = torch.Tensor([1]).to(device)

    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(
            input,
            F.one_hot(target, num_classes=input.size(-1)),
            self.y,
            reduction=self.reduction,
        )

        cent_loss = F.cross_entropy(
            F.normalize(input), target, reduction=self.reduction
        )
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss
