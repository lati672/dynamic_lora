from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader

from dynamic_lora.core.lora_app.config import TrainingConfig
from dynamic_lora.core.lora_app.data import collate_batch, prepare_example


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = REPO_ROOT / "artifacts" / "dynamic_lora" / "datasets"

AG_NEWS_LABELS = ("World", "Sports", "Business", "Sci/Tech")
YELP_LABELS = ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")
DBPEDIA_LABELS = (
    "Company",
    "Educational Institution",
    "Artist",
    "Athlete",
    "Office Holder",
    "Mean of Transportation",
    "Building",
    "Natural Place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written Work",
)


def select_subset(dataset: Dataset, sample_count: int, seed: int) -> Dataset:
    sample_count = max(1, min(sample_count, len(dataset)))
    if sample_count == len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_count))


def task_spec(task_name: str) -> dict:
    if task_name == "ag_news":
        return {
            "dataset_id": "ag_news",
            "text_field": "text",
            "train_split": "train",
            "eval_split": "test",
            "labels": AG_NEWS_LABELS,
            "question_template": (
                "Classify the following news article into exactly one category.\n"
                f"Categories: {', '.join(AG_NEWS_LABELS)}.\n\n"
                "Article: {text}\n\n"
                "Category:"
            ),
        }
    if task_name == "yelp_review_full":
        return {
            "dataset_id": "Yelp/yelp_review_full",
            "text_field": "text",
            "train_split": "train",
            "eval_split": "test",
            "labels": YELP_LABELS,
            "question_template": (
                "Read the following Yelp review and predict its star rating.\n"
                f"Possible ratings: {', '.join(YELP_LABELS)}.\n\n"
                "Review: {text}\n\n"
                "Rating:"
            ),
        }
    if task_name == "dbpedia_14":
        return {
            "dataset_id": "fancyzhx/dbpedia_14",
            "text_field": "content",
            "train_split": "train",
            "eval_split": "test",
            "labels": DBPEDIA_LABELS,
            "question_template": (
                "Classify the following encyclopedic passage into exactly one category.\n"
                f"Categories: {', '.join(DBPEDIA_LABELS)}.\n\n"
                "Passage: {text}\n\n"
                "Category:"
            ),
        }
    raise ValueError(f"Unsupported task: {task_name}")


def build_question(task_name: str, text: str) -> str:
    spec = task_spec(task_name)
    return spec["question_template"].format(text=text)


def label_text(task_name: str, label: int) -> str:
    spec = task_spec(task_name)
    return spec["labels"][int(label)]


def dataset_cache_dir(task_name: str, split_name: str, sample_count: int, seed: int) -> Path:
    return DATASET_CACHE_ROOT / task_name / f"{split_name}_count_{sample_count}_seed_{seed}"


def load_or_create_subset(task_name: str, split_name: str, sample_count: int, seed: int) -> Dataset:
    cache_dir = dataset_cache_dir(task_name, split_name, sample_count, seed)
    if cache_dir.exists():
        print(f"[data-cache] load task={task_name} split={split_name} path={cache_dir}")
        return load_from_disk(str(cache_dir))

    spec = task_spec(task_name)
    source_dataset = load_dataset(spec["dataset_id"], split=split_name)
    subset = select_subset(source_dataset, sample_count, seed)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(cache_dir))
    print(f"[data-cache] save task={task_name} split={split_name} path={cache_dir}")
    return subset


def load_task_datasets(
    task_name: str,
    train_count: int,
    eval_count: int,
    train_seed: int,
    eval_seed: int,
) -> tuple[Dataset, Dataset]:
    spec = task_spec(task_name)
    return (
        load_or_create_subset(task_name, spec["train_split"], train_count, train_seed),
        load_or_create_subset(task_name, spec["eval_split"], eval_count, eval_seed),
    )


def build_dataloader(
    tokenizer,
    config: TrainingConfig,
    task_name: str,
    source_dataset: Dataset,
    shuffle_seed: int | None = None,
) -> DataLoader:
    rows = source_dataset.map(
        lambda example: prepare_example(
            tokenizer=tokenizer,
            config=config,
            question_text=build_question(task_name, example[task_spec(task_name)["text_field"]]),
            answer_text=label_text(task_name, int(example["label"])),
            max_length=config.max_length,
        )
    )
    dataset = rows.remove_columns(source_dataset.column_names)
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(shuffle_seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_batch,
    )
