import torch

class ExplainableNLITrainer:
    def __init__(self, model, loss_fn, optimizer, device, warmup_epochs=1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.warmup_epochs = warmup_epochs

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        warmup = epoch < self.warmup_epochs
        total_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            loss_dict = self.loss_fn(outputs, batch, warmup=warmup)
            loss = loss_dict['Total loss']

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
