from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, mean_squared_error
from torch.utils.data import DataLoader

from training_v2 import CustomBert


def compute_metrics(labels, y_preds):
    pcl_threshold = 1.5
    pred_cl = y_preds > pcl_threshold
    true_cl = labels > pcl_threshold

    mse = mean_squared_error(labels, y_preds)
    acc = torch.mean((pred_cl == true_cl).float())
    f1p = f1_score(true_cl, pred_cl, pos_label=True)

    results = {"mse": mse, "acc": acc, "f1p": f1p}

    return results


@dataclass
class ForMetrics:
    label_ids: torch.Tensor
    predictions: torch.Tensor


def main():
    model_path = "results/model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomBert()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    dev = Dataset.load_from_disk("data/dev")

    dev.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_loader = DataLoader(dev, batch_size=16)

    global binary_classifier
    binary_classifier = False

    y_preds = []
    y_trues = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            y_pred = model(input_ids, attention_mask).cpu()
            labels = batch["labels"]

            y_preds.append(y_pred)
            y_trues.append(labels)

        metrics = compute_metrics(torch.cat(y_trues), torch.cat(y_preds))
        print(metrics)


if __name__ == "__main__":
    main()
