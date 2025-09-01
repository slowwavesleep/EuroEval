# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = [
#     "datasets==4.0.0",
#     "huggingface-hub==0.34.4",
#     "requests==2.32.5",
# ]
# ///

"""Create the EstQA dataset and upload to HF Hub."""

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from requests import HTTPError


def main() -> None:
    """Create the EstQA dataset and upload to HF Hub."""
    target_repo_id = "EuroEval/estqa"
    ds = load_dataset("TalTechNLP/EstQA")

    ds = ds.filter(
        lambda row: (len(row["answers"]) > 0) and (len(row["answers"][0]) >= 0)
    )

    df = ds["train"].to_pandas()

    # train and test are grouped differently
    # train needs to be converted to be consistent
    groups = []
    for id_value, group_df in df.groupby("id"):
        group_answers = group_df["answers"]
        answer_starts = [el[0]["answer_start"] for el in group_answers]
        texts = [el[0]["text"] for el in group_answers]
        cur_row = group_df.iloc[0].to_dict()
        cur_row["answers"] = {"answer_start": answer_starts, "text": texts}
        groups.append(cur_row)

    train_ds = Dataset.from_list(groups)  # 512 after grouping

    # 776, 603 - original sizes
    train_size = 384
    val_size = 128
    test_size = 603

    new_ds = DatasetDict()
    new_ds["train"] = train_ds.select(range(train_size))
    new_ds["val"] = train_ds.skip(train_size).select(range(val_size))
    new_ds["test"] = ds["test"].select(range(test_size))

    # remove the dataset from Hugging Face Hub if it already exists
    try:
        api: HfApi = HfApi()
        api.delete_repo(target_repo_id, repo_type="dataset")
    except HTTPError:
        pass

    # Push the dataset to the Hugging Face Hub
    new_ds.push_to_hub(target_repo_id, private=True)


if __name__ == "__main__":
    main()
