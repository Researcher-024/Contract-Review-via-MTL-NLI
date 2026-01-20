import torch
import random
from tqdm import tqdm
from rapidfuzz import fuzz, process

LABEL_MAP = {
    "entailment": 0,
    "contradiction": 1,
    "notmentioned": 2,
    "neutral": 2
}

def normalize_label(label: str) -> int:
    if not isinstance(label, str):
        raise ValueError(f"Invalid label format: {label}")

    key = label.strip().lower().replace(" ", "").replace("-", "")

    if key in LABEL_MAP:
        return LABEL_MAP[key]

    raise ValueError(f"Unknown decision label: {label}")

def fuzzy_find_span(context: str, evidence: str):
    """
    Attempts to locate evidence text inside the context.
    Returns (char_start, char_end).
    """

    if not isinstance(evidence, str) or not evidence.strip():
        return 0, 1

    ev = evidence.strip()

    # Exact match
    idx = context.find(ev)
    if idx != -1:
        return idx, idx + len(ev)

    # Fuzzy match
    match = process.extractOne(ev, [context], scorer=fuzz.partial_ratio)
    if match is None or match[1] <= 80:
        return 0, 1

    win = min(len(ev), len(context))
    best_score, best_idx = -1, 0

    for i in range(0, len(context) - win + 1):
        chunk = context[i:i + win]
        score = fuzz.ratio(chunk, ev)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_idx + win
  
@torch.no_grad()
def encode_evidence_text(
    encoder,
    tokenizer,
    text,
    device,
    max_length=128
):
    """
    Returns: Tensor[H]
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    hidden = encoder(**enc).last_hidden_state  # 1 x L x H
    mask = enc["attention_mask"].unsqueeze(-1)

    pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return pooled.squeeze(0).cpu()

def prepare_dataset(
    df,
    tokenizer,
    encoder,
    device,
    max_length=512,
    stride=128,
    max_neutral_windows=2,
    seed=42
):
    random.seed(seed)
    all_features = []

    # --------------------------------------------------
    # Precompute fallback evidence representation
    # --------------------------------------------------
    fallback_repr = encode_evidence_text(
        encoder,
        tokenizer,
        "Evidence not available",
        device
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):

        contract = str(row["contract-text"])
        hypothesis = str(row["hypothesis"])
        evidence = str(row["evidence-text"])
        orig_label = normalize_label(row["decision"])

        # --------------------------------------------------
        # Evidence representation target (document-level)
        # --------------------------------------------------
        if orig_label == 2 or not evidence.strip():
            target_evidence_repr = fallback_repr
        else:
            target_evidence_repr = encode_evidence_text(
                encoder,
                tokenizer,
                evidence,
                device
            )

        # --------------------------------------------------
        # Gold character span (for Loss 1)
        # --------------------------------------------------
        if orig_label == 2:
            char_start, char_end = 0, 1
        else:
            char_start, char_end = fuzzy_find_span(contract, evidence)
            char_start = max(0, min(char_start, len(contract) - 1))
            char_end = max(char_start + 1, min(char_end, len(contract)))

        tok = tokenizer(
            hypothesis,
            contract,
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            padding="max_length"
        )

        pos_windows, neg_windows = [], []

        for i in range(len(tok["input_ids"])):

            input_ids = tok["input_ids"][i]
            attention_mask = tok["attention_mask"][i]
            offsets = tok["offset_mapping"][i]
            seq_ids = tok.sequence_ids(i)

            start_pos, end_pos = 0, 0
            found = False
            label = orig_label

            if orig_label != 2:
                for t, (os, oe) in enumerate(offsets):
                    if seq_ids[t] != 1:
                        continue
                    if os <= char_start < oe:
                        start_pos = t
                    if os < char_end <= oe:
                        end_pos = t
                        found = True

                if not found:
                    start_pos, end_pos = 0, 0
                    label = 2

            feature = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": int(start_pos),
                "end_positions": int(end_pos),
                "labels": label,
                "target_evidence_representation": target_evidence_repr
            }

            if label == 2:
                neg_windows.append(feature)
            else:
                pos_windows.append(feature)

        # --------------------------------------------------
        # Window-level sampling
        # --------------------------------------------------
        all_features.extend(pos_windows)

        if len(neg_windows) > 0:
            keep_k = min(max_neutral_windows, len(neg_windows))
            all_features.extend(random.sample(neg_windows, keep_k))

    return all_features
