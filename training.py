import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import f1_score, mean_squared_error
from torch.nn.functional import mse_loss
from transformers import (
    DistilBertModel,
    Trainer,
    TrainingArguments,
)


class CustomBert(nn.Module):
    def __init__(self):
        super(CustomBert, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.regression = nn.Linear(self.distilbert.config.dim, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        y = self.regression(pooled_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss = mse_loss(labels, y)

        return y if labels is None else (loss, y)


def compute_metrics(pred):
    labels = pred.label_ids
    y = pred.predictions
    pred_cl = y > 1.5
    true_cl = labels > 1.5

    mse = mean_squared_error(labels, y)
    acc = np.mean(pred_cl == true_cl)
    f1 = f1_score(true_cl, pred_cl, pos_label=True)

    results = {"mse": mse, "acc": acc, "f1p": f1}

    return results


def main():
    torch.manual_seed(892)
    np.random.seed(892)

    train = Dataset.load_from_disk("data/train")
    val = Dataset.load_from_disk("data/val")
    dev = Dataset.load_from_disk("data/dev")

    model = CustomBert()

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(eval_result)

    torch.save(model.state_dict(), "results/model.pth")


if __name__ == "__main__":
    main()

    # Copy this for testing model
    # example = next(iter(train))
    # x, attention, y = example["input_ids"], example["attention_mask"], example["labels"]
    # x, attention, y = torch.tensor(x), torch.tensor(attention), torch.tensor(y)
    # x, attention, y = x.unsqueeze(0), attention.unsqueeze(0), y.unsqueeze(0)
    # pred = model(x, attention)
    # print(pred)
    # print(y)
    # print(pred.shape)
