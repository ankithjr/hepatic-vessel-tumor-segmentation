import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


def soft_erode_3d(img):
    p1 = -F.max_pool3d(
        -img,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(1, 0, 0)
    )

    p2 = -F.max_pool3d(
        -img,
        kernel_size=(1, 3, 1),
        stride=1,
        padding=(0, 1, 0)
    )

    p3 = -F.max_pool3d(
        -img,
        kernel_size=(1, 1, 3),
        stride=1,
        padding=(0, 0, 1)
    )

    return torch.min(torch.min(p1, p2), p3)


def soft_dilate_3d(img):
    return F.max_pool3d(
        img,
        kernel_size=3,
        stride=1,
        padding=1
    )


def soft_open_3d(img):
    return soft_dilate_3d(
        soft_erode_3d(img)
    )


def soft_skeletonize_3d(img, iterations=5):
    opened = soft_open_3d(img)
    skeleton = F.relu(img - opened)

    for _ in range(iterations):
        img = soft_erode_3d(img)
        opened = soft_open_3d(img)

        delta = F.relu(img - opened)

        skeleton = skeleton + F.relu(
            delta - skeleton * delta
        )

    return skeleton


class VesselClDiceLoss(nn.Module):

    def __init__(
        self,
        base_loss,
        cldice_weight=0.3,
        skeleton_iterations=5,
        smooth=1e-5
    ):
        super().__init__()

        self.base_loss = base_loss
        self.cldice_weight = cldice_weight
        self.skeleton_iterations = skeleton_iterations
        self.smooth = smooth

    def forward(self, logits, target):

        base = self.base_loss(
            logits,
            target
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        vessel_prediction = probabilities[:, 1:2]

        if target.ndim == logits.ndim:
            target_labels = target[:, 0]
        else:
            target_labels = target

        vessel_target = (
            target_labels == 1
        ).float().unsqueeze(1)

        vessel_prediction_small = F.avg_pool3d(
            vessel_prediction,
            kernel_size=2,
            stride=2
        )

        vessel_target_small = F.max_pool3d(
            vessel_target,
            kernel_size=2,
            stride=2
        )

        pred_skeleton = soft_skeletonize_3d(
            vessel_prediction_small,
            self.skeleton_iterations
        )

        target_skeleton = soft_skeletonize_3d(
            vessel_target_small,
            self.skeleton_iterations
        )

        tprec = (
            (pred_skeleton * vessel_target_small).sum()
            + self.smooth
        ) / (
            pred_skeleton.sum()
            + self.smooth
        )

        tsens = (
            (target_skeleton * vessel_prediction_small).sum()
            + self.smooth
        ) / (
            target_skeleton.sum()
            + self.smooth
        )

        cldice = (
            2.0 * tprec * tsens
        ) / (
            tprec + tsens + self.smooth
        )

        cldice_loss = 1.0 - cldice

        return (
            base
            + self.cldice_weight * cldice_loss
        )


class nnUNetTrainerClDice(nnUNetTrainer):

    def _build_loss(self):

        base_loss = DC_and_CE_loss(
            {
                "batch_dice":
                    self.configuration_manager.batch_dice,

                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        loss = VesselClDiceLoss(
            base_loss=base_loss,
            cldice_weight=0.3,
            skeleton_iterations=5
        )

        if self.enable_deep_supervision:

            deep_supervision_scales = (
                self._get_deep_supervision_scales()
            )

            weights = np.array([
                1 / (2 ** i)
                for i in range(
                    len(deep_supervision_scales)
                )
            ])

            if (
                self.is_ddp
                and not self._do_i_compile()
            ):
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            loss = DeepSupervisionWrapper(
                loss,
                weights
            )

        return loss

