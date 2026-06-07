import json
import os
import subprocess
from pathlib import Path

import optuna

PROJECT = "uncertainty-v5"

DATASET_NAME = "agnews"
DATASET_NUM_CLS = 4
DATASET_MODE = "long_tail"
DATASET_IR = 10
SEED = 50

EXPERIMENT = f"mixbelief_bedl_newidea_{DATASET_MODE}_ir{DATASET_IR}_{DATASET_NAME}"
STUDY_NAME = f"mixbelief_newidea_{DATASET_MODE}_ir{DATASET_IR}_{DATASET_NAME}"


def objective(trial):
    # --- search space ---
    mu_prior = trial.suggest_float("loss.mu_prior", 0.5, 1.0, step=0.05)
    anneal = trial.suggest_int("loss.annealing_step", 5, 10)
    switch_epoch = trial.suggest_int("train.switch_epoch", 2, 5)
    mix_alpha = trial.suggest_float("mix.alpha", 0.1, 1.0, step=0.1)

    trial_id = trial.number

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["TRIAL_ID"] = str(trial_id)
    env["PYTHONUNBUFFERED"] = "1"

    nproc_per_node = int(env.get("NPROC_PER_NODE", "3"))

    launcher_log_dir = Path("results") / "optuna" / "_launcher_logs" / STUDY_NAME
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    stdout_file = launcher_log_dir / f"trial{trial_id}_stdout.log"
    stderr_file = launcher_log_dir / f"trial{trial_id}_stderr.log"

    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc_per_node}",
        "main_optuna_test.py",
        f"dataset.name={DATASET_NAME}",
        f"dataset.num_cls={DATASET_NUM_CLS}",
        f"dataset.mode={DATASET_MODE}",
        f"dataset.ir={DATASET_IR}",
        f"train.seed={SEED}",
        "train.eval_interval=1000000",
        "train.distributed=true",
        "train.curriculum=true",
        "train.uncertainty=true",
        "mix.method=mix-belief",
        f"mix.alpha={mix_alpha}",
        "loss.type=CE",
        "train.epochs=10",
        f"train.switch_epoch={switch_epoch}",
        f"loss.mu_prior={mu_prior}",
        f"loss.annealing_step={anneal}",
        "hydra.run.dir=.",
        "hydra.job.chdir=false",
        f"experiment={EXPERIMENT}",
        "output.result_dir=results/optuna",
        "output.model_dir=models_save/optuna",
    ]

    with open(stdout_file, "w") as out, open(stderr_file, "w") as err:
        p = subprocess.run(
            cmd,
            env=env,
            stdout=out,
            stderr=err,
            text=True,
        )

    if p.returncode != 0:
        tail = stderr_file.read_text(errors="ignore")[-4000:]
        raise RuntimeError(
            f"Trial {trial_id} failed with return code {p.returncode}.\n"
            f"Last stderr lines:\n{tail}"
        )

    metrics_path = (
        Path("results")
        / "optuna"
        / DATASET_NAME
        / f"{EXPERIMENT}_trial{trial_id}"
        / f"seed{SEED}"
        / "last_val_metrics.json"
    )

    if not metrics_path.exists():
        raise RuntimeError(f"Metrics file not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    score = float(metrics["best_val_f1"])

    return score


def main():
    storage = f"sqlite:///{os.environ.get('OPTUNA_DB_PATH', f'{STUDY_NAME}.db')}"

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,
            n_ei_candidates=64,
            multivariate=True,
            seed=42,
        ),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, timeout=int(7 * 3600))


if __name__ == "__main__":
    main()
