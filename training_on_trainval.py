import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import mse_loss
from transformers import (
    DistilBertModel,
    Trainer,
    TrainingArguments,
)

import wandb


class CustomBert(nn.Module):
    def __init__(self, transformer_out=range(4, 7), dropout=0.1, class_weights=None):
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
        # for module in self.distilbert.modules():
        #     if isinstance(module, torch.nn.Dropout):
        #         module.p = dropout

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
        if layer is not None and layer < 0:
            return

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


def main(
    label="",
    dropout=0.1,
    transformer_out=range(4, 7),
    binary_flag=False,
    unfreeze_layers=[-1, 5, 4, 3, 2, 1, 0],
    unfreeze_epochs=[3, 3, 3, 3, 3, 3, 3],
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    **kwargs,
):

    run_name = f"{label}_d_{dropout}"
    run_id = wandb.util.generate_id()
    wandb.init(project="distilbert", name=run_name, id=run_id)

    # if doing a sweep, then batch_size and learning_rate are passed in the config
    try:
        learning_rate = wandb.config.learning_rate
        per_device_train_batch_size = wandb.config.batch_size
        dropout = wandb.config.dropout
    except:
        pass

    torch.manual_seed(892)
    np.random.seed(892)
    random.seed(892)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(892)

    train = Dataset.load_from_disk("data/train")
    val = Dataset.load_from_disk("data/val")
    dev = Dataset.load_from_disk("data/dev")

    # concatenate datasets to get train_val
    # had to convert one of the column types or it throws an issue
    val_cast = val.cast(train.features)
    train_val = concatenate_datasets([train, val_cast])

    # Preprocessing was adjusted to use the score as the labels
    # If we don't want to use the score then we have to go back to using the pcl as the labels
    global binary_classifier
    binary_classifier = binary_flag
    if binary_classifier:
        train = train_val.map(lambda x: {"score": x["labels"]})
        train = train_val.map(lambda x: {"labels": x["pcl"]})

    labels = train_val["labels"]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights)
    compute_metrics_weighted = partial(compute_metrics, class_weights=class_weights)

    model = CustomBert(
        dropout=dropout,
        class_weights=class_weights.to(device),
        transformer_out=transformer_out,
    )
    model.to(device)

    model.freeze()
    for i, (layer, num_epochs) in enumerate(zip(unfreeze_layers, unfreeze_epochs)):
        if i != 0:
            model.unfreeze(layer)
            wandb.init(project="distilbert", id=run_id, resume="must")

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=16,
            learning_rate=learning_rate,
            warmup_steps=419,
            evaluation_strategy="epoch",
            report_to="wandb",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_val,
            eval_dataset=dev,
            compute_metrics=compute_metrics_weighted,
        )

        trainer.train()
        wandb.finish()

    wandb.init(project="distilbert", name=run_name, id=run_id, resume="must")
    config = wandb.config
    config.dropout = dropout
    config.transformer_out = transformer_out
    config.binary_flag = binary_flag
    for k, v in kwargs.items():
        setattr(config, k, v)
    # for k, v in eval_result.items():
    #     setattr(config, k, v)
    config.unfreeze_layers = unfreeze_layers
    config.unfreeze_epochs = unfreeze_epochs
    config.learning_rate = learning_rate
    config.batch_size = per_device_train_batch_size

    torch.save(model.state_dict(), "results/model.pth")
    wandb.finish()


if __name__ == "__main__":

    unfreeze_layers = [-1, 5, 4, 3, 2, 1, 0]
    unfreeze_epochs = [3, 3, 3, 3, 3, 3, 3]

    main(
        label="ht_t&v",
        unfreeze_layers=unfreeze_layers,
        unfreeze_epochs=unfreeze_epochs,
        dropout=0.10877582740940311,
        per_device_train_batch_size=8,
        learning_rate=0.00001834940916078444,
    )
