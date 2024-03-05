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


def load_model():
    model_path = "results/model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomBert()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    return model, device


def load_data(path):
    dev = Dataset.load_from_disk(path)
    dev.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_loader = DataLoader(dev, batch_size=16)

    return dev_loader


def predict(model, device, loader):
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            y_pred = model(input_ids, attention_mask).cpu()
            labels = batch["labels"]

            y_preds.append(y_pred)
            y_trues.append(labels)

    return torch.cat(y_preds), torch.cat(y_trues)


def load_test(path):
    test = Dataset.load_from_disk(path)
    test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_loader = DataLoader(test, batch_size=16)

    return test_loader


def predict_test(model, device, loader):
    y_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            y_pred = model(input_ids, attention_mask).cpu()

            y_preds.append(y_pred)

    return torch.cat(y_preds)


def main():
    model, device = load_model()

    dev_loader = load_data("data/dev")

    global binary_classifier
    binary_classifier = False

    y_preds, y_trues = predict(model, device, dev_loader)

    metrics = compute_metrics(y_trues, y_preds)
    print(metrics)
    pred_labels = (y_preds > 1.5).int()
    print(len(pred_labels))
    print(f1_score((y_trues > 1.5).int(), pred_labels, pos_label=True))

    with open("dev.txt", "w") as f:
        for pred in pred_labels:
            f.write(f"{pred.item()}\n")

    test_loader = load_test("data/test")
    y_preds = predict_test(model, device, test_loader)
    pred_labels = (y_preds > 1.5).int()
    with open("test.txt", "w") as f:
        for pred in pred_labels:
            f.write(f"{pred.item()}\n")


if __name__ == "__main__":
    main()
