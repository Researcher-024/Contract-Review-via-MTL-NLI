import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

def compute_class_weights(features, num_classes=3):
    labels = [f["labels"] for f in features]
    counts = Counter(labels)

    total = sum(counts.values())
    weights = []

    for i in range(num_classes):
        if counts[i] == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * counts[i]))

    weights = torch.tensor(weights, dtype=torch.float)
    return weights

class ExplainableNLILoss(nn.Module):
    def __init__(
        self,
        span_weight=0.9,
        decision_weight=1.0,
        repr_weight=0.85,
        coupling_weight=0.9,
        class_weights=None
    ):
        super().__init__()

        self.span_weight = span_weight
        self.decision_weight = decision_weight
        self.repr_weight = repr_weight
        self.coupling_weight = coupling_weight

        self.ce_span = nn.CrossEntropyLoss()

        # Decision loss with optional class weights
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.ce_decision = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.ce_decision = nn.CrossEntropyLoss()

    def forward(self, outputs, batch, warmup=False):
        # batch --> Input batch (contains y_gold), outputs --> output batch (contains y_preds)
        start_logits = outputs["start_logits"]   # (B, T)
        end_logits = outputs["end_logits"]       # (B, T)
        decision_logits = outputs["decision_logits"]  # (B, C)
        context_mask = outputs["context_mask"]   # (B, T) in {0,1}

        # --------------------------------------------------
        # Loss 1: Span loss
        # --------------------------------------------------
        L_span = (
            self.ce_span(start_logits, batch["start_positions"]) +
            self.ce_span(end_logits, batch["end_positions"])
        )

        # --------------------------------------------------
        # Loss 2: Decision loss (class-weighted)
        # --------------------------------------------------
        L_cls = self.ce_decision(
            decision_logits,
            batch["labels"]
        )

        loss = self.span_weight * L_span + self.decision_weight * L_cls

        if not warmup:
            # --------------------------------------------------
            # Loss 3: Evidence representation cosine loss
            # --------------------------------------------------
            pred = F.normalize(outputs["evidence_representation"], dim=-1)
            tgt = F.normalize(batch["target_evidence_representation"], dim=-1)
            L_repr = (1 - (pred * tgt).sum(dim=-1)).mean()

            # --------------------------------------------------
            # Loss 4: Normalized entropy–decision coupling
            # --------------------------------------------------

            # Masked softmax
            masked_start_logits = start_logits.masked_fill(context_mask == 0, -1e9)
            masked_end_logits = end_logits.masked_fill(context_mask == 0, -1e9)

            p_start = F.softmax(masked_start_logits, dim=-1)
            p_end = F.softmax(masked_end_logits, dim=-1)

            # Per-example entropy
            eps = 1e-8
            H_start = -(p_start * torch.log(p_start + eps)).sum(dim=-1)
            H_end = -(p_end * torch.log(p_end + eps)).sum(dim=-1)
            H_span = H_start + H_end   # (B,)

            # Normalize by maximum possible entropy
            valid_token_counts = context_mask.sum(dim=-1).float().clamp(min=1.0)
            H_max = 2.0 * torch.log(valid_token_counts + eps)  # start + end

            H_norm = H_span / H_max    # in [0, 1]

            # Entropy–decision coupling (one-way)
            L_couple = (H_norm * L_cls.detach()).mean()

            loss += (
                self.repr_weight * L_repr +
                self.coupling_weight * L_couple
            )
            loss_dict = {'Total loss': loss, 'Span loss': L_span, 'Decision loss': L_cls, 'Couple loss': L_couple, 'Repr loss': L_repr}
            return loss_dict
        else:
            loss_dict = {'Total loss': loss, 'Span loss': L_span, 'Decision loss': L_cls}
            return loss_dict
