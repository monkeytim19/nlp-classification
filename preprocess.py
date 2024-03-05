import os
import random
import shutil

import inflect
import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from nlpaug.flow import Sequential, Sometimes
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
    paragraphs["text"] = paragraphs["text"].fillna("")

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


def load_test_tsv():
    # Import main data file
    paragraphs_path = "data/task4_test.tsv"
    paragraphs = pd.read_csv(paragraphs_path, sep="\t", skiprows=0, header=None)
    paragraphs.columns = [
        "par_id",
        "art_id",
        "keyword",
        "country_code",
        "text",
    ]

    return paragraphs


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


def augment_text(df, augs, upsample, back_translate=True):
    if upsample > 0:
        pcl_df = df[df["pcl"] == 1].copy()
        with_replacement = True if upsample > 1 else False
        pcl_df = pcl_df.sample(
            frac=upsample, random_state=861, replace=with_replacement
        )
        sample_text = pcl_df["text"].iloc[0]
        print(f"Original: {sample_text}")

        aug_text = list(pcl_df["text"])
        for aug in augs:
            aug_text = aug.augment(aug_text)
        pcl_df["text"] = aug_text
        augmented_text = pcl_df["text"].iloc[0]
        print(f"Augmented: {augmented_text}")
    else:
        pcl_df = pd.DataFrame(columns=df.columns)

    if back_translate:
        back_translate_aug = naw.BackTranslationAug(
            from_model_name="Helsinki-NLP/opus-mt-en-zh",
            to_model_name="Helsinki-NLP/opus-mt-zh-en",
            max_length=1000,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        for_translation = df[df["pcl"] == 1].copy()
        print(f"Original: {for_translation.text.iloc[0]}")

        for_translation_text = list(for_translation["text"])
        translated_text = back_translate_aug.augment(for_translation_text)
        for_translation["text"] = translated_text
        print(f"Translated: {for_translation.text.iloc[0]}")
    else:
        for_translation = pd.DataFrame(columns=df.columns)

    return pd.concat([df, pcl_df, for_translation])


def keywords_to_stopwords(df):
    keywords = list(df.keyword.unique())

    # common issue is that the keyword is singular but the word in the text is plural or vice-versa
    # so add the singular and plural forms of the keyword to the stopwords
    p = inflect.engine()
    stopwords = set()
    for word in keywords:
        stopwords.add(word)
        stopwords.add(p.plural(word))
        singular = p.singular_noun(word)
        if singular:  # This is False when word is already singular
            stopwords.add(singular)

    return list(stopwords)


def preprocess(
    dir_path,
    sub_p=0.2,
    ins_p=0.05,
    upsample=2.0,
    use_stopwords=True,
    back_translate=True,
    preprompt="patronizing: ",
    postprompt="",
):
    train_val_df, dev_df = load_tsv()
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, random_state=861, stratify=train_val_df["labels"]
    )

    np.random.seed(861)
    random.seed(861)

    if use_stopwords:
        stopwords = keywords_to_stopwords(train_val_df)
    else:
        stopwords = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    substitute_aug = naw.ContextualWordEmbsAug(
        model_path="distilbert-base-uncased",
        action="substitute",
        aug_p=sub_p,
        aug_min=0,
        aug_max=1000,
        stopwords=stopwords,
        device=device,
    )
    insert_aug = naw.ContextualWordEmbsAug(
        model_path="distilbert-base-uncased",
        action="insert",
        aug_p=ins_p,
        aug_min=0,
        aug_max=1000,
        stopwords=stopwords,
        device=device,
    )

    augs = Sequential([substitute_aug, insert_aug])
    aug_train_df = augment_text(train_df, augs, upsample, back_translate)

    for df in [train_df, val_df, dev_df, aug_train_df]:
        df["text"] = preprompt + df["text"] + postprompt

    train = tokenise_df(aug_train_df)
    val = tokenise_df(val_df)
    dev = tokenise_df(dev_df)

    # to avoid tokenising every time, save down the datasets
    train.save_to_disk(f"{dir_path}/train")
    val.save_to_disk(f"{dir_path}/val")
    dev.save_to_disk(f"{dir_path}/dev")


def preprocess_test():
    test_df = load_test_tsv()
    test_df["text"] = "patronizing: " + test_df["text"]
    test = tokenise_df(test_df)
    test.save_to_disk("data/test")


if __name__ == "__main__":
    preprocess_test()

    # preprocess(
    #     dir_path="data",
    #     upsample=0.0,
    #     back_translate=False,
    #     preprompt="patronizing: ",
    # )

    # preprocess(
    #     dir_path="data",
    #     upsample=2.0,
    #     back_translate=True,
    #     preprompt="",
    # )

    # --------------------------------------------
    # For loading into the main file
    # train = Dataset.load_from_disk("data/train")
    # val = Dataset.load_from_disk("data/val")
    # dev = Dataset.load_from_disk("data/dev")
