import shutil
from pathlib import Path

import pandas as pd
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import Dataset, load_from_disk

from .data_configs import DATASETS
from .utils import create_synthetic_imbalance

SPLIT_SEED = 42
VALID_IRS = [10, 25, 50, 80, 100]
VALID_SRS = [0.1, 0.25, 0.5, 0.8]


def read_txtfile(path: str) -> pd.DataFrame:
    labels = []
    texts = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            label, text = line.split("\t", 1)
            labels.append(label)
            texts.append(text)
    return pd.DataFrame({"text": texts, "label": labels})


def _read_file(path: str, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return pd.read_json(f, lines=True)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "txt":
        return read_txtfile(path)
    raise ValueError(f"unknown format of file: {fmt}")


def preprocess_train_df(df, cfg):
    mode = cfg.dataset.mode
    if mode == "original":
        return df
    if mode == "long_tail" or mode == "step":
        if cfg.dataset.ir not in VALID_IRS and cfg.dataset.ir != 1:
            raise ValueError(
                f"IR must be one of {VALID_IRS} (or 1 for no imbalance), got ir={cfg.dataset.ir}"
            )

        if cfg.dataset.ir == 1:
            return df
        return create_synthetic_imbalance(
            df, cfg.dataset.ir, SPLIT_SEED, imbalance_type=cfg.dataset.mode
        )
    """if mode == "sample":
        if cfg.dataset.sr != 1.0 and cfg.dataset.sr not in VALID_SRS:
            raise ValueError(
                f"Sampling ratio must be 1.0 (no sampling) or one of {sorted(VALID_SRS)}, "
                f"but got sr={cfg.dataset.sr}"
            )
        if cfg.dataset.sr == 1.0:
            return df
        return create_sample(df, cfg.dataset.sr, SPLIT_SEED)"""
    raise ValueError(f"Unknown mode: {mode}")


def tokenize_df(df, tokenizer, cfg):
    ds = Dataset.from_pandas(df)
    return ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.train.max_length,
        ),
        batched=True,
    ).with_format("torch", columns=["input_ids", "attention_mask", "label"])


def load_raw_dfs(name: str):
    cfg = DATASETS[name]
    text_col = cfg["text_col"]
    label_col = cfg["label_col"]
    dfs = {}

    for split, path in cfg["files"].items():
        df = _read_file(path, cfg["format"])
        df = df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )

        df["text"] = df["text"].astype(str)
        # df['label'] = df['label'].astype(int)
        dfs[split] = df

    # si cfg['splits'] est défini, on splitte manuellement
    # TODO prendre val a partir du train et non pas a partir de test
    if cfg.get("need_label_encoding", True):
        available = [dfs[s] for s in ("train", "val", "test") if s in dfs]
        all_labels = pd.concat(available, axis=0, ignore_index=True)["label"]
        le = LabelEncoder().fit(all_labels)
        for split in dfs:
            dfs[split]["label"] = le.transform(dfs[split]["label"]).astype(int)
    else:
        for split in dfs:
            dfs[split]["label"] = dfs[split]["label"].astype(int)

    if "splits" in cfg and "train" in dfs and "test" in dfs:
        train_p, val_p = (p for _, p in cfg["splits"])
        full_train = dfs.pop("train")
        train_df, val_df = train_test_split(
            full_train,
            test_size=val_p / (train_p + val_p),
            stratify=full_train.label,
            random_state=SPLIT_SEED,
        )
        dfs = {"train": train_df, "val": val_df, "test": dfs["test"]}

    elif "splits" in cfg and "all" in dfs:
        full = dfs.pop("all")
        train_p, val_p, test_p = (p for _, p in cfg["splits"])
        train_df, temp = train_test_split(
            full, test_size=(1 - train_p), stratify=full.label, random_state=SPLIT_SEED
        )
        val_df, test_df = train_test_split(
            temp,
            test_size=test_p / (val_p + test_p),
            stratify=temp.label,
            random_state=SPLIT_SEED,
        )
        dfs = {"train": train_df, "val": val_df, "test": test_df}
    return dfs["train"], dfs["val"], dfs["test"]


def _cache_is_complete(cache_dir: Path) -> bool:
    """
    Vérifie que le cache HuggingFace Dataset est complet.
    """
    required_splits = ["train", "val", "test"]

    for split in required_splits:
        split_dir = cache_dir / split

        if not split_dir.exists():
            return False

        # Fichiers typiques créés par Dataset.save_to_disk()
        if not (split_dir / "dataset_info.json").exists():
            return False

        if not (split_dir / "state.json").exists():
            return False

    return True


def build_dataset(cfg, tokenizer, logger):
    # ------------------------------------------------------------------
    # 1. Detect DistributedDataParallel context
    # ------------------------------------------------------------------
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_main_process = rank == 0
    else:
        rank = 0
        world_size = 1
        is_main_process = True

    # ------------------------------------------------------------------
    # 2. Build cache directory
    # ------------------------------------------------------------------
    if cfg.dataset.mode in ["long_tail", "step"]:
        cache_dir = (
            Path("cache") / f"{cfg.dataset.name}_{cfg.dataset.mode}_{cfg.dataset.ir}"
        )

    elif cfg.dataset.mode == "sample":
        cache_dir = (
            Path("cache") / f"{cfg.dataset.name}_{cfg.dataset.mode}_{cfg.dataset.sr}"
        )

    elif cfg.dataset.mode == "original":
        cache_dir = Path("cache") / f"{cfg.dataset.name}_{cfg.dataset.mode}"

    else:
        raise ValueError(f"Unknown dataset mode: {cfg.dataset.mode}")

    # ------------------------------------------------------------------
    # 3. Create/tokenize/cache dataset only on rank 0
    # ------------------------------------------------------------------
    if is_main_process:
        cache_complete = _cache_is_complete(cache_dir)

        if not cache_complete:
            logger.warning(
                f"Dataset cache is missing or incomplete at {cache_dir}. "
                f"Rebuilding it on rank 0."
            )

            # Important : supprimer le cache incomplet avant reconstruction
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            cache_dir.mkdir(parents=True, exist_ok=True)

            train_df, val_df, test_df = load_raw_dfs(name=cfg.dataset.name)

            logger.info(
                f"label values count of train_df before preprocessing: "
                f"{train_df.label.value_counts()}"
            )
            logger.info(
                f"label values count of val_df before preprocessing: "
                f"{val_df.label.value_counts()}"
            )
            logger.info(
                f"label values count of test_df before preprocessing: "
                f"{test_df.label.value_counts()}"
            )

            train_df = preprocess_train_df(train_df, cfg)

            logger.info(
                f"label values count of train_df after preprocessing: "
                f"{train_df.label.value_counts()}"
            )

            train_ds = tokenize_df(train_df, tokenizer, cfg)
            val_ds = tokenize_df(val_df, tokenizer, cfg)
            test_ds = tokenize_df(test_df, tokenizer, cfg)

            train_ds.save_to_disk(cache_dir / "train")
            val_ds.save_to_disk(cache_dir / "val")
            test_ds.save_to_disk(cache_dir / "test")

            logger.info(f"Dataset saved to cache: {cache_dir}")

        else:
            logger.info(f"Using existing complete dataset cache: {cache_dir}")

    # ------------------------------------------------------------------
    # 4. Wait until rank 0 finishes creating the cache
    # ------------------------------------------------------------------
    if distributed:
        dist.barrier()

    # ------------------------------------------------------------------
    # 5. All ranks load the same cached datasets
    # ------------------------------------------------------------------
    if not _cache_is_complete(cache_dir):
        raise FileNotFoundError(
            f"Dataset cache is still incomplete after preprocessing: {cache_dir}. "
            f"Expected train/, val/, and test/ subdirectories."
        )

    train_ds = load_from_disk(cache_dir / "train")
    val_ds = load_from_disk(cache_dir / "val")
    test_ds = load_from_disk(cache_dir / "test")

    # ------------------------------------------------------------------
    # 6. DistributedSampler for training
    # ------------------------------------------------------------------
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    else:
        train_sampler = None

    # ------------------------------------------------------------------
    # 7. Build DataLoaders
    # ------------------------------------------------------------------
    num_workers = getattr(cfg.train, "num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
