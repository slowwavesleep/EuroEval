# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the Estonian valence dataset and upload to HF Hub."""

from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the Estonian valence dataset and upload to HF Hub."""
    target_repo_id = "EuroEval/winogrande_et"

    # start from the official source
    human_ds = load_dataset("tartuNLP/winogrande_et", "human_translated")
    mt_ds = load_dataset("tartuNLP/winogrande_et", "machine_translated")

    # target split sizes
    train_size = 1024
    val_size = 256
    test_size = 2048

    # we don't have human translations for train and dev
    ds = DatasetDict(
        {
            "train": mt_ds["train"].select(range(train_size)),
            "validation": mt_ds["dev"].select(range(val_size)),
            "test": human_ds["test"].select(range(min(test_size, len(human_ds["test"])))),
        }
    )

    # please don't share the answers explicitly though
    ds["test"] = ds["test"].map(lambda row: {"answer": row["qID"][-1]})

    try:
        api = HfApi()
        api.delete_repo(target_repo_id, repo_type="dataset")
    except HTTPError:
        pass

    ds.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
