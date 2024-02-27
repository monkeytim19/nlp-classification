import os
import shutil

import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer


def load_tsv():
    # Import main data file
    paragraphs_path = "data/dontpatronizeme_pcl.tsv"
    paragraphs = pd.read_csv(paragraphs_path, sep="\t", skiprows=4, header=None)
    paragraphs.columns = [
        "par_id",
        "art_id",
        "keyword",
        "country_code",
        "text",
        "labels",
    ]
    PCL_threshold = 2
    paragraphs["pcl"] = (paragraphs["labels"] >= PCL_threshold).astype(int)
    paragraphs["labels"] = paragraphs["labels"].astype(float)
    paragraphs = paragraphs.dropna()  # paragraph 8640 has None text

    # Split into train+val and dev
    train_split_path = "data/train_semeval_parids-labels.csv"
    dev_split_path = "data/dev_semeval_parids-labels.csv"
    train_ids = pd.read_csv(train_split_path)
    dev_ids = pd.read_csv(dev_split_path)
    dev_ids.columns = ["par_id", "label_category_vector"]
    train_ids.columns = ["par_id", "label_category_vector"]
    train_val = paragraphs.merge(train_ids, on="par_id", how="inner")
    dev = paragraphs.merge(dev_ids, on="par_id", how="inner")

    # Note at this point we have 3 label columns - need to be careful about which ones get into training
    # labels (the score), label_category_vector, pcl (the true classification label)
    return train_val, dev


def tokenise_df(df):
    tokeniser = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def encode(dataset):
        # For efficiency, we truncate to a standard length of 512 tokens
        # In the train dataset, exactly 2 paragraphs have length > 512 (559 and 999)
        # We consider it is not a material issue to truncate these 2 paragraphs
        # But must make a note in the report about patronising language after 512 tokens
        return tokeniser(
            dataset["text"], padding="max_length", max_length=512, truncation=True
        )

    dataset = Dataset.from_pandas(df)
    encoded = dataset.map(encode)

    return encoded


def augment_text(df, aug, upsample=1.0):
    pcl_df = df[df["pcl"] == 1].copy()
    pcl_df = pcl_df.sample(frac=upsample, random_state=861)
    orig_text = list(pcl_df["text"])
    aug_text = aug.augment(orig_text)
    pcl_df["text"] = aug_text

    sample_text = df["text"].iloc[0]
    print(f"Original: {sample_text}")
    print(f"Type of sample text: {type(sample_text)}")
    augmented_text = aug.augment(sample_text)
    print(f"Augmented: {augmented_text}")
    print(f"Type of augmented text: {type(augmented_text)}")

    return pd.concat([df, pcl_df])


def preprocess(aug_p=0.0):
    train_val_df, dev_df = load_tsv()
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=861, stratify=train_val_df["labels"]
    )

    np.random.seed(861)
    aug = naw.ContextualWordEmbsAug(
        model_path="distilbert-base-uncased", action="substitute", aug_p=aug_p
    )
    aug_train_df = augment_text(train_df, aug, upsample=1)

    train = tokenise_df(aug_train_df)
    val = tokenise_df(val_df)
    dev = tokenise_df(dev_df)

    # to avoid tokenising every time, save down the datasets. delete the old datasets first.
    for path in ["data/train", "data/val", "data/dev"]:
        if os.path.exists(path):
            shutil.rmtree(path)
    train.save_to_disk("data/train")
    val.save_to_disk("data/val")
    dev.save_to_disk("data/dev")


if __name__ == "__main__":
    preprocess(0.0)

    # --------------------------------------------
    # For loading into the main file
    # train = Dataset.load_from_disk("data/train")
    # val = Dataset.load_from_disk("data/val")
    # dev = Dataset.load_from_disk("data/dev")
