import pandas as pd
from datasets import Dataset
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
        "label_score",
    ]
    PCL_threshold = 2
    paragraphs["label"] = (paragraphs["label_score"] >= PCL_threshold).astype(int)
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
    # label, label_score, label_category_vector
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


if __name__ == "__main__":
    train_val_df, dev_df = load_tsv()
    train_val = tokenise_df(train_val_df)
    dev = tokenise_df(dev_df)

    train, val = train_val.train_test_split(test_size=0.2, seed=861).values()

    # to avoid tokenising every time, save down the datasets
    train.save_to_disk("data/train")
    val.save_to_disk("data/val")
    dev.save_to_disk("data/dev")

    # --------------------------------------------
    # For loading into the main file
    # train = Dataset.load_from_disk("data/train")
    # val = Dataset.load_from_disk("data/val")
    # dev = Dataset.load_from_disk("data/dev")
