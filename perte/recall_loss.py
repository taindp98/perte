import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallLoss(nn.Module):
    """ An unofficial implementation of
        <Recall Loss for Imbalanced Image Classification
        and Semantic Segmentation>
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        recall = TP / (TP + FN)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *],
        for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for
        Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """

    def __init__(self, weight=None, device=torch.device("cpu")):
        super(RecallLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5
        self.device = device

    def forward(self, input, target):
        N, C = input.size()[:2]
        input = F.softmax(input, 1)
        _, predict = torch.max(input, 1)  # # (N, C, *) ==> (N, 1, *)

        predict = predict.view(N, 1, -1)  # (N, 1, *)
        target = target.view(N, 1, -1)  # (N, 1, *)
        last_size = target.size(-1)

        predict_onehot = torch.zeros(
            (N, C, last_size), device=self.device
        )  # (N, 1, *) ==> (N, C, *)
        predict_onehot.scatter_(1, predict, 1)  # (N, C, *)
        target_onehot = torch.zeros(
            (N, C, last_size), device=self.device
        )  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        true_positive = torch.sum(predict_onehot * target_onehot, dim=2)  # (N, C)
        total_target = torch.sum(target_onehot, dim=2)  # (N, C)
        tp = true_positive + self.smooth
        tp_fn = total_target + self.smooth
        recall = tp / tp_fn  # (N, C)

        if hasattr(self, "weight"):
            if self.weight.type() != input.type():
                self.weight = self.weight.type_as(input)
                recall = recall * self.weight * C  # (N, C)
        recall_loss = 1 - torch.mean(recall)  # 1

        return recall_loss
