import torch
import argparse
import torch.nn.functional as F

@torch.no_grad()
def run_chunked_inference(
    model,
    tokenizer,
    contract_text: str,
    hypothesis: str,
    max_length: int = 512,
    stride: int = 128,
    max_span_length: int = 128,
    device: torch.device = None
):
    """
    Returns a list of candidate explanations across chunks.
    """
    ID2LABEL = {0:'entailment', 1:'contradiction', 2:'not mentioned'}
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ------------------------------------------------------------
    # Tokenize with overflow (chunking)
    # ------------------------------------------------------------
    enc = tokenizer(
        hypothesis,
        contract_text,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"]
    seq_ids = [enc.sequence_ids(i) for i in range(len(input_ids))]

    results = []

    # ------------------------------------------------------------
    # Process each chunk independently
    # ------------------------------------------------------------
    for i in range(len(input_ids)):

        outputs = model(
            input_ids=input_ids[i].unsqueeze(0),
            attention_mask=attention_mask[i].unsqueeze(0)
        )

        start_logits = outputs["start_logits"][0]
        end_logits = outputs["end_logits"][0]
        decision_logits = outputs["decision_logits"][0]
        context_mask = outputs["context_mask"][0]

        # Decision
        probs = F.softmax(decision_logits, dim=-1)
        pred_id = probs.argmax().item()
        confidence = probs[pred_id].item()

        decision_label = ID2LABEL[pred_id]

        # # Skip neutral / not-mentioned
        # if pred_id == 2:
        #     continue

        # --------------------------------------------------------
        # Evidence span decoding (same as single-window inference)
        # --------------------------------------------------------
        start_logits = start_logits.masked_fill(context_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(context_mask == 0, -1e9)

        p_start = F.softmax(start_logits, dim=-1)
        p_end = F.softmax(end_logits, dim=-1)

        L = start_logits.size(0)
        span_scores = torch.triu(p_start.unsqueeze(1) * p_end.unsqueeze(0))

        # Enforce max span length
        if max_span_length is not None:
            for s in range(L):
                span_scores[s, s + max_span_length :] = 0.0

        flat_idx = span_scores.view(-1).argmax().item()
        start_idx = flat_idx // L
        end_idx = flat_idx % L

        # --------------------------------------------------------
        # Token span → character span → text
        # --------------------------------------------------------
        char_start, char_end = None, None
        for t in range(start_idx, end_idx + 1):
            if seq_ids[i][t] != 1:
                continue
            os, oe = offsets[i][t]
            if os == 0 and oe == 0:
                continue
            if char_start is None:
                char_start = os
            char_end = oe

        if char_start is None or char_end is None:
            evidence_text = None
        else:
            evidence_text = contract_text[char_start:char_end].strip()

        results.append({
            "decision": decision_label,
            "evidence_text": evidence_text,
            "confidence": confidence,
            "chunk_index": i
        })
    return results

def ParseArgs():
    """
    Parse command-line arguments for model configuration.
    Examples:
      python mycode.py --model_type base --dataset_type cnli --model_version 3
      python mycode.py -m large -d cuad -v 1
    """
    parser = argparse.ArgumentParser(description="Run MultiTaskLegalNLI with a chosen model/dataset/version.")
    parser.add_argument(
        "-m", "--model_type",
        required=True,
        choices=["base", "large"],
        help="Model size/type to use (e.g., base or large)."
    )
    parser.add_argument(
        "-d", "--dataset_type",
        required=True,
        choices=["cnli", "cuad", "control"],
        help="Dataset identifier."
    )
    parser.add_argument(
        "-v", "--model_version",
        required=True,
        type=int,
        help="Model version number (e.g., 1, 2, 3...)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    import torch
    import pandas as pd
    from MultiTaskLegalNLI import MultiTaskLegalNLI
    from transformers import RobertaTokenizerFast, RobertaModel

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cli
    args = ParseArgs()
    model_type = args.model_type #base or large
    dataset_type = args.dataset_type #cnli or cuad or control
    model_version = args.model_version # 1 or 2 or 3...

    # save information
    MODEL_CKPT = f'Roberta_{model_type}_{dataset_type}_{model_version}.pt'
    MODEL_LOAD_PATH = f'/workspace/data/Aditya/MTL/Models/{MODEL_CKPT}'
    TEST_FILE_NAME = 'Contract-NLI-test.xlsx'
    TEST_FILE_PATH = f'/workspace/data/Aditya/MTL/Data/{TEST_FILE_NAME}'
    SAVE_RESULTS_PATH = f'/workspace/data/Aditya/MTL/Data/{MODEL_CKPT.split(".")[0]}_version{model_version}.xlsx'

    # load encoder and tokenizer
    MODEL_ID = f'FacebookAI/roberta-{model_type}'
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)
    encoder = RobertaModel.from_pretrained(MODEL_ID).to(device)

    # initialize model and load state dict
    model = MultiTaskLegalNLI(encoder=encoder, tokenizer=tokenizer).to(device)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=True))

    # load test data
    test_df = pd.read_excel(TEST_FILE_PATH, engine='openpyxl')

    # run inference on each row
    test_df['Predicted Results'] = None
    for idx, row in test_df.iterrows():
        contract_text = row['contract-text']
        hypothesis = row['hypothesis']

        results = run_chunked_inference(
            model=model,
            tokenizer=tokenizer,
            contract_text=contract_text,
            hypothesis=hypothesis,
            device=device
        )
        test_df.loc[idx, 'Predicted Results'] = results
    
    # save results
    test_df.to_excel(SAVE_RESULTS_PATH, engine='openpyxl', index=False)
    print(f"Inference completed. Results saved to {SAVE_RESULTS_PATH}")
