from prepare_dataset import prepare_dataset
from LegalNLIDataset import LegalNLIDataset
from MultiTaskLegalNLI import MultiTaskLegalNLI
from ExplainableNLILoss import ExplainableNLILoss, compute_class_weights
from trainer import ExplainableNLITrainer

if __name__ == "__main__":
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    # from transformers import RobertaTokenizerFast, RobertaModel
    # from transformers import AutoTokenizer, AutoModel

    try:
        # set training parameters and paths
        ckpt = 'Deberta_base_cnli_1' #'Roberta_large_cnli_2' #'Roberta_base_cnli_1'
        train_file = 'Contract-NLI-train.xlsx'
        BATCH_SIZE = 10  # size: 10 for roberta-base | 5 for roberta-large
        EPOCHS = 90 #50 # total epochs 100 for roberta-base
        WARMUP_EPOCHS = 40 # warm-up epochs 40 for roberta-base
        TRAIN_FILE_PATH = f'/workspace/data/Aditya/MTL/Data/{train_file}'
        # MODEL_ID = 'microsoft/deberta-base' #'FacebookAI/roberta-large' # 'FacebookAI/roberta-base'
        MODEL_ID = 'FacebookAI/roberta-large'
        MODEL_SAVE_PATH = f'/workspace/data/Aditya/MTL/Models/{ckpt}.pt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load and initialize data, model, tokenizer and prepare dataset features.
        train_df = pd.read_excel(TRAIN_FILE_PATH, engine='openpyxl').sample(frac=1).reset_index(drop=True)
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)
        encoder = RobertaModel.from_pretrained(MODEL_ID).to(device)
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        # encoder = AutoModel.from_pretrained(MODEL_ID).to(device)
        features = prepare_dataset(df=train_df, tokenizer=tokenizer, encoder=encoder, device=device)

        # prepare dataset and dataloader
        ds = LegalNLIDataset(features)
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        # initialize model
        model = MultiTaskLegalNLI(encoder=encoder, tokenizer=tokenizer).to(device)

        # compute class weights
        class_weights = compute_class_weights(features)
        
        # initialize optimizer, loss function and trainer
        loss_criterion = ExplainableNLILoss(class_weights=class_weights).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        Trainer = ExplainableNLITrainer(model=model, loss_fn=loss_criterion, optimizer=optimizer, device=device, warmup_epochs=WARMUP_EPOCHS)

        # training loop
        for epoch in range(EPOCHS):
            loss = Trainer.train_epoch(dataloader, epoch)
            print(f"Epoch {epoch}: loss = {loss:.4f}")
        
        # save the trained model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Training completed. Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Following error occurred: {e}")
