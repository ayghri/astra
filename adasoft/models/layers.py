# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
import torch
import torch.nn.functional as F
from torch import nn


class KLDiv(nn.Module):
    def forward(self, student_logits, teacher_logits, temperature=1.0):
        inputs = F.log_softmax(student_logits / temperature, dim=-1)
        targets = F.log_softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(inputs, targets, reduction="batchmean", log_target=True)


class TopPCrossEntropy(nn.Module):
    """Cross-entropy on teacher's top-p probability mass.

    This computes H(p_t_top_p, p_s) where p_t_top_p is the teacher
    distribution truncated to the smallest prefix of classes whose
    cumulative probability >= top_p, then renormalized. The student
    distribution remains over all classes; only teacher mass outside
    top-p is ignored in the loss.

    Args:
        top_p (float): cumulative probability threshold in (0, 1].
        temperature (float): softmax temperature applied to both logits.
    """

    def __init__(self, top_p: float = 0.9, temperature: float = 1.0):
        super().__init__()
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        self.top_p = top_p
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """Return cross-entropy between teacher_top_p and full student dist.

        Shapes:
            student_logits: (batch, num_classes)
            teacher_logits: (batch, num_classes)
        """
        t = self.temperature

        # Softmax distributions
        teacher_log_probs = F.log_softmax(teacher_logits / t, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits / t, dim=-1)

        # Define top-p support based on teacher distribution
        sorted_probs, sorted_indices = torch.sort(
            teacher_probs, dim=-1, descending=True
        )
        cumprobs = sorted_probs.cumsum(dim=-1)

        top_p_mask_sorted = cumprobs <= self.top_p
        # Ensure at least one element per row (include highest-prob class)
        top_p_mask_sorted[..., 0] = True

        batch_size, num_classes = teacher_probs.shape
        top_p_mask = teacher_probs.new_zeros(
            (batch_size, num_classes), dtype=torch.bool
        )
        top_p_mask.scatter_(dim=-1, index=sorted_indices, src=top_p_mask_sorted)

        # Mask teacher distribution and renormalize on top-p support
        masked_teacher_probs = teacher_probs * top_p_mask
        teacher_mass = masked_teacher_probs.sum(dim=-1, keepdim=True)

        one = torch.ones_like(teacher_mass)
        safe_teacher_mass = torch.where(teacher_mass > 0, teacher_mass, one)

        masked_teacher_probs = torch.where(
            teacher_mass > 0,
            masked_teacher_probs / safe_teacher_mass,
            teacher_probs,
        )

        # Cross-entropy H(p_t_top_p, p_s) = - sum p_t_top_p * log p_s
        ce = -(masked_teacher_probs * student_log_probs).sum(dim=-1)
        return ce.mean()


class TopPKLDiv(nn.Module):
    """KL divergence using teacher's top-p probability mass.

    Computes KL(student || teacher_top_p), where teacher_top_p is the
    teacher distribution truncated to the smallest prefix of classes
    whose cumulative probability >= top_p, then renormalized. The
    student distribution is over all classes; teacher mass outside
    top-p is ignored when defining the target distribution.

    Args:
        top_p (float): cumulative probability threshold in (0, 1].
        temperature (float): softmax temperature applied to both logits.
    """

    def __init__(self, top_p: float = 0.9, temperature: float = 1.0):
        super().__init__()
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        self.top_p = top_p
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """Return KL(student || teacher_top_p).

        Shapes:
            student_logits: (batch, num_classes)
            teacher_logits: (batch, num_classes)
        """
        t = self.temperature

        # Softmax distributions
        teacher_log_probs = F.log_softmax(teacher_logits / t, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits / t, dim=-1)

        # Define top-p support based on teacher distribution
        sorted_probs, sorted_indices = torch.sort(
            teacher_probs, dim=-1, descending=True
        )
        cumprobs = sorted_probs.cumsum(dim=-1)

        top_p_mask_sorted = cumprobs <= self.top_p
        top_p_mask_sorted[..., 0] = True  # ensure at least one element

        batch_size, num_classes = teacher_probs.shape
        top_p_mask = teacher_probs.new_zeros(
            (batch_size, num_classes), dtype=torch.bool
        )
        top_p_mask.scatter_(dim=-1, index=sorted_indices, src=top_p_mask_sorted)

        # Mask teacher distribution and renormalize on top-p support
        masked_teacher_probs = teacher_probs * top_p_mask
        teacher_mass = masked_teacher_probs.sum(dim=-1, keepdim=True)

        one = torch.ones_like(teacher_mass)
        safe_teacher_mass = torch.where(teacher_mass > 0, teacher_mass, one)

        masked_teacher_probs = torch.where(
            teacher_mass > 0,
            masked_teacher_probs / safe_teacher_mass,
            teacher_probs,
        )

        masked_teacher_log_probs = torch.log(masked_teacher_probs + 1e-12)

        # KL(student || teacher_top_p)
        return F.kl_div(
            student_log_probs,
            masked_teacher_log_probs,
            reduction="batchmean",
            log_target=True,
        )
