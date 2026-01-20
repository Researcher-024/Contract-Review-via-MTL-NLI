import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidenceIdentificationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU()
        )
        self.span_out = nn.Linear(hidden_size // 2, 2)

    def forward(self, hidden_states):
        span_logits = self.span_out(self.proj(hidden_states))
        return span_logits[..., 0], span_logits[..., 1]


class MultiTaskLegalNLI(nn.Module):
    def __init__(
        self,
        encoder,
        tokenizer,
        num_classes=3,
        unfreeze_last_n=2,
        disable_dropout_in_frozen=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.tokenizer = tokenizer
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        H = encoder.config.hidden_size

        self.evidence_head = EvidenceIdentificationHead(H)

        self.classifier = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(H, num_classes)
        )

        # Freeze / unfreeze encoder
        layers = encoder.encoder.layer
        for p in encoder.parameters():
            p.requires_grad = False

        n = min(unfreeze_last_n, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        if disable_dropout_in_frozen:
            for layer in layers[:-n]:
                layer.eval()

    def _build_masks(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device

        sep_positions = (input_ids == self.sep_id).nonzero(as_tuple=False)
        hyp_mask = torch.zeros((B, L), device=device)
        ctx_mask = torch.zeros((B, L), device=device)

        for b in range(B):
            seps = sep_positions[sep_positions[:, 0] == b][:, 1]
            hyp_mask[b, 1:seps[0]] = 1
            ctx_mask[b, seps[1] + 1 :] = 1

        ctx_mask *= (input_ids != self.pad_id)
        return hyp_mask, ctx_mask

    @staticmethod
    def soft_span_pooling(hidden, start_logits, end_logits, context_mask):
        start_logits = start_logits.masked_fill(context_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(context_mask == 0, -1e9)

        p_start = F.softmax(start_logits, dim=-1)
        p_end = F.softmax(end_logits, dim=-1)

        span_scores = torch.triu(p_start.unsqueeze(2) * p_end.unsqueeze(1))
        token_weights = span_scores.sum(dim=2) + span_scores.sum(dim=1)
        token_weights = token_weights / (token_weights.sum(dim=1, keepdim=True) + 1e-8)

        return torch.bmm(token_weights.unsqueeze(1), hidden).squeeze(1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        hyp_mask, ctx_mask = self._build_masks(input_ids)

        start_logits, end_logits = self.evidence_head(hidden)

        hyp_repr = (hidden * hyp_mask.unsqueeze(-1)).sum(dim=1)
        hyp_repr /= (hyp_mask.sum(dim=1, keepdim=True) + 1e-8)

        evidence_repr = self.soft_span_pooling(
            hidden, start_logits, end_logits, ctx_mask
        )

        decision_logits = self.classifier(
            torch.cat([hyp_repr, evidence_repr], dim=-1)
        )

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "decision_logits": decision_logits,
            "evidence_representation": evidence_repr,
            'context_mask': ctx_mask
        }
