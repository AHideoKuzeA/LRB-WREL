import torch
import torch.nn as nn
import torch.nn.functional as F

def sharpen_probs(logits, T: float = 0.7, dim: int = 1):
    return F.softmax(logits / T, dim=dim)

def masked_mean(x, mask, eps: float = 1e-8):
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

class ConsistencyKLLoss(nn.Module):
    def __init__(self, temperature_t: float = 0.7, temperature_s: float = 1.0, reduction="mean"):
        super().__init__()
        self.Tt = temperature_t
        self.Ts = temperature_s
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits, conf_mask: torch.Tensor):
        with torch.no_grad():
            p_t = sharpen_probs(teacher_logits, T=self.Tt, dim=1).clamp_min(1e-8)

        log_p_s = F.log_softmax(student_logits / self.Ts, dim=1)
        kl = F.kl_div(log_p_s, p_t, reduction='none')
        kl = kl.sum(dim=1)
        if self.reduction == "mean":
            return masked_mean(kl, conf_mask)
        elif self.reduction == "sum":
            while conf_mask.dim() < kl.dim():
                conf_mask = conf_mask.unsqueeze(1)
            return (kl * conf_mask).sum()
        else:
            return kl

class WeakSemiSupervisedLoss(nn.Module):
    def __init__(self, sup_weight_dice: float = 0.1,
                 unsup_weight: float = 0.6,
                 thr_init: float = 0.6, thr_final: float = 0.8, epochs_ramp: int = 5,
                 temperature_teacher: float = 0.7, temperature_student: float = 1.0,
                 class_axis: int = 1):
        super().__init__()
        from loss import Loss as SupervisedLoss
        self.sup_loss = SupervisedLoss(weight=sup_weight_dice)
        self.cons_loss = ConsistencyKLLoss(temperature_t=temperature_teacher,
                                           temperature_s=temperature_student)
        self.lam_u_base = unsup_weight
        self.thr_init = thr_init
        self.thr_final = thr_final
        self.epochs_ramp = max(1, epochs_ramp)
        self.class_axis = class_axis

    def _schedule_tau(self, epoch: int):
        p = min(1.0, epoch / float(self.epochs_ramp))
        return self.thr_init + p * (self.thr_final - self.thr_init)

    def _schedule_lambda_u(self, epoch: int):
        p = min(1.0, epoch / float(self.epochs_ramp))
        return self.lam_u_base * (0.3 + 0.7 * p)

    @torch.no_grad()
    def _confidence_mask(self, teacher_logits, tau: float):
        probs = F.softmax(teacher_logits, dim=self.class_axis)
        maxprob, _ = probs.max(dim=self.class_axis)
        return (maxprob >= tau).float()

    def forward(self,
                student_logits_sup, target_sup,
                student_logits_weak=None, teacher_logits_weak=None,
                epoch: int = 0):
        L_sup = self.sup_loss(student_logits_sup, target_sup)

        if (student_logits_weak is not None) and (teacher_logits_weak is not None):
            tau = self._schedule_tau(epoch)
            lam_u = self._schedule_lambda_u(epoch)
            conf_mask = self._confidence_mask(teacher_logits_weak, tau=tau)
            L_cons = self.cons_loss(student_logits_weak, teacher_logits_weak, conf_mask)
            return L_sup + lam_u * L_cons, {"L_sup": L_sup.item(), "L_cons": L_cons.item(),
                                            "tau": float(tau), "lambda_u": float(lam_u)}
        else:
            return L_sup, {"L_sup": L_sup.item()}
