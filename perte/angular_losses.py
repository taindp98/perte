import torch
import torch.nn.functional as F
from torch import nn
import math


class AngularPenaltySMLoss(nn.Module):
    def __init__(
        self,
        loss_type,
        in_fts,
        out_fts,
        class_weights=None,
        s=None,
        m=None,
        eps=1e-7,
        device=torch.device("cpu"),
    ):
        """
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        """
        super(AngularPenaltySMLoss, self).__init__()
        # loss_type = loss_type.lower()
        # assert loss_type in  ['arcface', 'sphereface', 'cosface']
        self.loss_type = loss_type

        if self.loss_type == "arcface":
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        if self.loss_type == "sphereface":
            self.s = 30.0 if not s else s
            self.m = 1.35 if not m else m
        if self.loss_type == "cosface":
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        if self.loss_type == "acloss":
            self.s = 30.0 if not s else s
            self.m = 0.3 if not m else m
        self.eps = eps
        self.device = device
        self.cls_weight = class_weights
        self.fc = nn.Linear(in_fts, out_fts, bias=False, device=device)

    def forward(self, input, target):
        """
        input shape (N, in_features)
        """
        input = input.to(self.device)
        target = target.to(self.device)
        self.fc = self.fc.to(self.device)

        assert len(input) == len(target)
        assert torch.min(target) >= 0

        input = F.normalize(input, p=2, dim=1)

        for w in self.fc.parameters():
            w = F.normalize(w, p=2, dim=1)

        input = self.fc(input)

        if self.loss_type == "cosface":
            numerator = self.s * (
                torch.diagonal(input.transpose(0, 1)[target]) - self.m
            )
        if self.loss_type == "arcface":
            numerator = self.s * torch.cos(
                torch.acos(
                    torch.clamp(
                        torch.diagonal(input.transpose(0, 1)[target]),
                        -1.0 + self.eps,
                        1 - self.eps,
                    )
                )
                + self.m
            )
        if self.loss_type == "sphereface":
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(
                    torch.clamp(
                        torch.diagonal(input.transpose(0, 1)[target]),
                        -1.0 + self.eps,
                        1 - self.eps,
                    )
                )
            )
        if self.loss_type == "acloss":
            acos = (
                torch.acos(
                    torch.clamp(
                        torch.diagonal(input.transpose(0, 1)[target]),
                        -1.0 + self.eps,
                        1 - self.eps,
                    )
                )
                + self.m
            )
            numerator = self.s * g_theta(acos)
        excl = torch.cat(
            [
                torch.cat((input[i, :y], input[i, y + 1 :])).unsqueeze(0)
                for i, y in enumerate(target)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        if self.cls_weight:
            self.w = torch.tensor(
                [self.cls_weight[i] for i in target], device=self.device
            )
            L = self.w * (numerator - torch.log(denominator))
        else:
            L = numerator - torch.log(denominator)

        return -torch.mean(L)


def g_theta(arccos, k=0.3):
    numerator_1 = 1 + math.exp(-math.pi / 2.0 / k)
    denominator_1 = 1 - math.exp(-math.pi / 2.0 / k)
    sigmoid1 = numerator_1 / denominator_1
    numerator_2 = 1 - torch.exp(arccos / k - math.pi / 2.0 / k)
    denominator_2 = 1 + torch.exp(arccos / k - math.pi / 2.0 / k)
    sigmoid2 = numerator_2 / denominator_2
    cos_t = sigmoid1 * sigmoid2
    return cos_t
