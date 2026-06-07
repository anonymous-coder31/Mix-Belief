import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig

from src.mix_belief_trainer_new import MixBeliefTrainer2
from src.trainers import Trainer


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base="1.1", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    base = hydra.utils.get_original_cwd()

    out_dir = os.path.join(
        base, cfg.output.result_dir, cfg.dataset.name, cfg.experiment
    )
    os.makedirs(out_dir, exist_ok=True)

    results_file = os.path.join(out_dir, "_results.txt")
    log_file = os.path.join(out_dir, "main.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("=== Start Experiment ===")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Experiment: {cfg.experiment}")
    logger.info(f"Output directory: {out_dir}")

    results = {}

    for seed in cfg.train.seeds:
        cfg.train.seed = seed

        logger.info(f"--- Run for seed = {cfg.train.seed} ---")

        if cfg.train.curriculum:
            trainer = MixBeliefTrainer2(cfg)
            logger.info("Curriculum training selected.")
        else:
            trainer = Trainer(cfg)
            logger.info("Simple training selected.")

        metrics = trainer.run()

        results[int(seed)] = metrics

        logger.info(f"Finished seed {cfg.train.seed}.")
        logger.info(f"Metrics: {metrics}")

    logger.info("=== End of runs ===")

    save_mean_std_results(results, results_file)

    logger.info(f"Aggregated results saved to: {results_file}")

    return results


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def save_mean_std_results(results, result_file, label_names=None):
    """
    Save scalar metrics and per-class metrics over multiple seeds.

    Expected results format:
    {
        seed1: {
            "acc": ...,
            "macro_f1": ...,
            "tail_recall": ...,
            "ece": ...,
            "per_class_precision": np.array([...]),
            "per_class_recall": np.array([...]),
            "per_class_f1": np.array([...]),
            "per_class_support": np.array([...]),
            "cm": np.array([[...], ...])
        },
        seed2: {...},
        ...
    }

    Args:
        results: dict, metrics per seed.
        result_file: path to output txt file.
        label_names: optional list of class names.
                     Example: ["negative", "somewhat_negative", "neutral", ...]
    """

    # ------------------------------------------------------------------
    # 1. Identify scalar metrics
    # ------------------------------------------------------------------
    non_scalar_metrics = {
        "cm",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "per_class_support",
        "classification_report",
    }

    metric_names = set()
    for _, metrics in results.items():
        metric_names.update(metrics.keys())

    scalar_metric_names = sorted(
        [name for name in metric_names if name not in non_scalar_metrics]
    )

    # ------------------------------------------------------------------
    # 2. Identify number of classes
    # ------------------------------------------------------------------
    n_classes = None

    for _, metrics in results.items():
        per_class_f1 = _to_numpy(metrics.get("per_class_f1", None))
        if per_class_f1 is not None:
            n_classes = len(per_class_f1)
            break

    if n_classes is not None:
        if label_names is None:
            label_names = [f"class_{i}" for i in range(n_classes)]
        else:
            assert len(label_names) == n_classes, (
                f"label_names length={len(label_names)} but n_classes={n_classes}"
            )

    # ------------------------------------------------------------------
    # 3. Write results
    # ------------------------------------------------------------------
    with open(result_file, "a") as f:
        f.write("=" * 100 + "\n")
        f.write("RESULTS OVER SEEDS\n")
        f.write("=" * 100 + "\n\n")

        # ==============================================================
        # A. Scalar metrics per seed
        # ==============================================================
        f.write("-" * 100 + "\n")
        f.write("SCALAR METRICS PER SEED\n")
        f.write("-" * 100 + "\n\n")

        scalar_rows = []

        for seed, metrics in results.items():
            row = {"seed": seed}

            for metric_name in scalar_metric_names:
                value = metrics.get(metric_name, None)

                # Ignore arrays/lists/dicts
                if isinstance(value, (list, tuple, dict, np.ndarray)):
                    continue

                value = _safe_float(value)
                if value is not None:
                    row[metric_name] = value

            scalar_rows.append(row)

        scalar_df = pd.DataFrame(scalar_rows)

        if not scalar_df.empty:
            f.write(scalar_df.to_string(index=False))
            f.write("\n\n")

        # ==============================================================
        # B. Scalar metrics mean and std
        # ==============================================================
        f.write("-" * 100 + "\n")
        f.write("SCALAR METRICS MEAN ± STD OVER SEEDS\n")
        f.write("-" * 100 + "\n\n")

        summary_rows = []

        for metric_name in scalar_metric_names:
            values = []

            for _, metrics in results.items():
                value = metrics.get(metric_name, None)

                if isinstance(value, (list, tuple, dict, np.ndarray)):
                    continue

                value = _safe_float(value)
                if value is not None and not np.isnan(value):
                    values.append(value)

            if len(values) == 0:
                continue

            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

            summary_rows.append(
                {
                    "metric": metric_name,
                    "values": values,
                    "mean": mean_value,
                    "std": std_value,
                    "mean±std": f"{mean_value:.6f} ± {std_value:.6f}",
                }
            )

        summary_df = pd.DataFrame(summary_rows)

        if not summary_df.empty:
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")

        # ==============================================================
        # C. Per-class metrics per seed
        # ==============================================================
        if n_classes is not None:
            f.write("-" * 100 + "\n")
            f.write("PER-CLASS METRICS PER SEED\n")
            f.write("-" * 100 + "\n\n")

            per_class_rows = []

            for seed, metrics in results.items():
                per_class_precision = _to_numpy(
                    metrics.get("per_class_precision", None)
                )
                per_class_recall = _to_numpy(metrics.get("per_class_recall", None))
                per_class_f1 = _to_numpy(metrics.get("per_class_f1", None))
                per_class_support = _to_numpy(metrics.get("per_class_support", None))

                if (
                    per_class_precision is None
                    or per_class_recall is None
                    or per_class_f1 is None
                ):
                    continue

                for class_id in range(n_classes):
                    row = {
                        "seed": seed,
                        "class_id": class_id,
                        "precision": float(per_class_precision[class_id]),
                        "recall": float(per_class_recall[class_id]),
                        "f1": float(per_class_f1[class_id]),
                    }

                    if per_class_support is not None:
                        row["support"] = int(per_class_support[class_id])

                    per_class_rows.append(row)

            per_class_df = pd.DataFrame(per_class_rows)

            if not per_class_df.empty:
                f.write(per_class_df.to_string(index=False))
                f.write("\n\n")

            # ==========================================================
            # D. Per-class metrics mean and std over seeds
            # ==========================================================
            f.write("-" * 100 + "\n")
            f.write("PER-CLASS METRICS MEAN ± STD OVER SEEDS\n")
            f.write("-" * 100 + "\n\n")

            per_class_summary_rows = []

            for class_id in range(n_classes):
                for metric_name in ["precision", "recall", "f1"]:
                    values = (
                        per_class_df.loc[
                            per_class_df["class_id"] == class_id, metric_name
                        ]
                        .dropna()
                        .values
                    )

                    if len(values) == 0:
                        continue

                    mean_value = float(np.mean(values))
                    std_value = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    )

                    per_class_summary_rows.append(
                        {
                            "class_id": class_id,
                            "metric": metric_name,
                            "values": list(values),
                            "mean": mean_value,
                            "std": std_value,
                            "mean±std": f"{mean_value:.6f} ± {std_value:.6f}",
                        }
                    )

                # Support is usually identical across seeds if test set is fixed.
                # Still, we report mean/std for completeness.
                if "support" in per_class_df.columns:
                    support_values = (
                        per_class_df.loc[
                            per_class_df["class_id"] == class_id, "support"
                        ]
                        .dropna()
                        .values
                    )

                    if len(support_values) > 0:
                        support_mean = float(np.mean(support_values))
                        support_std = (
                            float(np.std(support_values, ddof=1))
                            if len(support_values) > 1
                            else 0.0
                        )

                        per_class_summary_rows.append(
                            {
                                "class_id": class_id,
                                "metric": "support",
                                "values": list(support_values),
                                "mean": support_mean,
                                "std": support_std,
                                "mean±std": f"{support_mean:.2f} ± {support_std:.2f}",
                            }
                        )

            per_class_summary_df = pd.DataFrame(per_class_summary_rows)

            if not per_class_summary_df.empty:
                f.write(per_class_summary_df.to_string(index=False))
                f.write("\n\n")

        # ==============================================================
        # E. Confusion matrix
        # ==============================================================
        cms = []

        for _, metrics in results.items():
            cm = _to_numpy(metrics.get("cm", None))
            if cm is not None:
                cms.append(cm)

        if len(cms) > 0:
            f.write("-" * 100 + "\n")
            f.write("CONFUSION MATRIX SUMMED OVER SEEDS\n")
            f.write("-" * 100 + "\n\n")

            cm_sum = np.sum(cms, axis=0)
            f.write(str(cm_sum))
            f.write("\n\n")

            f.write("-" * 100 + "\n")
            f.write("CONFUSION MATRIX MEAN OVER SEEDS\n")
            f.write("-" * 100 + "\n\n")

            cm_mean = np.mean(cms, axis=0)
            f.write(str(np.round(cm_mean, 2)))
            f.write("\n\n")

        f.write("\n\n")


if __name__ == "__main__":
    results = main()
