from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import mse_loss
from transformers import (
    DistilBertModel,
    Trainer,
    TrainingArguments,
)


class CustomBert(nn.Module):
    def __init__(self, transformer_out=6, dropout=0.1, class_weights=None):
        super(CustomBert, self).__init__()
        # Instead of just using the output of the final hidden layer,
        # you can also pass in a range of hidden layers to concatenate their outputs
        self.transformer_out = (
            range(transformer_out, transformer_out + 1)
            if isinstance(transformer_out, int)
            else transformer_out
        )
        out_dim = len(self.transformer_out) * 768

        # Use pretrained DistilBert. Force it to use our dropout
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", output_hidden_states=True
        )  # type: DistilBertModel
        for module in self.distilbert.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout

        # Then apply a dense hidden layer down to 768, and a final layer down to 1
        self.feedforward = nn.Sequential(
            nn.Linear(out_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 1),
        )

        if class_weights is not None:
            self.class_weights = class_weights
            self.pos_weight = class_weights[1] / class_weights[0]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Recommended pooling approach for DistilBert is to average over the hidden state sequence
        # instead of outputs.last_hidden_state[:, 0], which is used for Bert which uses [CLS] token
        pooled_output = []
        for i in self.transformer_out:
            hs = outputs.hidden_states[i]
            mask = attention_mask.unsqueeze(-1)
            hs = hs * mask
            mean_hs = hs.sum(dim=1) / mask.sum(dim=1)
            pooled_output.append(mean_hs)

        # We also concatenate the outputs of multiple layers if chosen by the user
        cat_output = torch.cat(pooled_output, dim=1)

        # Apply dense feedforward
        y = self.feedforward(cat_output).squeeze(-1)

        # Outside the Trainer, we return the predictions
        if labels is None:
            return y

        # Inside the Trainer, we also need to return the loss
        global binary_classifier
        if binary_classifier:
            loss = F.binary_cross_entropy_with_logits(
                y, labels, pos_weight=self.pos_weight
            )
        else:
            loss = mse_loss(y, labels, reduction="none")
            weights = self.class_weights[labels.long()]
            loss = loss * weights
            loss = loss.mean()
        return loss, y

    def freeze(self):
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze(self, layer=None):
        for name, param in self.distilbert.named_parameters():
            if layer is None or name.startswith(f"transformer.layer.{layer}"):
                param.requires_grad = True


def compute_metrics(pred, class_weights=None):
    labels = pred.label_ids
    y = pred.predictions
    global binary_classifier
    pcl_threshold = 0.5 if binary_classifier else 1.5
    pred_cl = y > pcl_threshold
    true_cl = labels > pcl_threshold

    mse = mean_squared_error(labels, y)
    acc = np.mean(pred_cl == true_cl)
    f1p = f1_score(true_cl, pred_cl, pos_label=True)

    results = {"mse": mse, "acc": acc, "f1p": f1p}

    if class_weights is not None:
        weights = class_weights[labels]
        mse_weighted = mean_squared_error(labels, y, sample_weight=weights)
        results["mse_weighted"] = mse_weighted

    return results


def main(dropout=0.1, transformer_out=6, binary_flag=False):
    torch.manual_seed(892)
    np.random.seed(892)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = Dataset.load_from_disk("data/train")
    val = Dataset.load_from_disk("data/val")
    dev = Dataset.load_from_disk("data/dev")

    # Preprocessing was adjusted to use the score as the labels
    # If we don't want to use the score then we have to go back to using the pcl as the labels
    global binary_classifier
    binary_classifier = binary_flag
    if binary_classifier:
        train = train.map(lambda x: {"score": x["labels"]})
        train = train.map(lambda x: {"labels": x["pcl"]})

    labels = train["labels"]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights)
    compute_metrics_weighted = partial(compute_metrics, class_weights=class_weights)

    model = CustomBert(class_weights=class_weights.to(device)).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics_weighted,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(eval_result)

    torch.save(model.state_dict(), "results/model.pth")


if __name__ == "__main__":
    # for dropout in [0, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #     main(dropout)
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
