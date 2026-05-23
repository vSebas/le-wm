import json
import logging
import os
import re
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import pyarrow as pa

if not hasattr(pa, "PyExtensionType"):
    pa.PyExtensionType = pa.ExtensionType

import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from hdf5_dataset import HDF5Dataset
from module import SIGReg
from utils import (
    BestModelObjectCallback,
    EpochMetricsCallback,
    ModelObjectCallBack,
    get_column_normalizer,
    get_img_preprocessor,
)


def _load_training_dataset(dataset_cfg, transform=None):
    dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR") or dataset_cfg.pop("cache_dir", None)
    dataset_path = Path(dataset_name).expanduser()

    if cache_dir is not None and not dataset_path.exists():
        cache_dir = Path(cache_dir).expanduser()
        for candidate in (
            cache_dir / dataset_name,
            cache_dir / f"{dataset_name}.h5",
            cache_dir / f"{dataset_name}.hdf5",
        ):
            if candidate.exists():
                dataset_path = candidate
                break

    if dataset_path.suffix.lower() in {".h5", ".hdf5"} and dataset_path.exists():
        return HDF5Dataset(name=str(dataset_path), transform=transform, **dataset_cfg)

    if cache_dir is not None:
        dataset_cfg["cache_dir"] = str(cache_dir)

    import stable_worldmodel as swm

    return swm.data.load_dataset(dataset_name, transform=transform, **dataset_cfg)


class EpisodeSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, episode_indices):
        self.dataset = dataset
        self.episode_indices = set(int(ep) for ep in episode_indices)
        self.indices = [
            idx
            for idx, (ep_idx, _) in enumerate(dataset.clip_indices)
            if int(ep_idx) in self.episode_indices
        ]
        if not self.indices:
            raise ValueError("EpisodeSubset has no clips. Check the split and sequence length.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def _episode_row_indices(dataset, episode_indices):
    rows = [
        np.arange(dataset.offsets[ep], dataset.offsets[ep] + dataset.lengths[ep])
        for ep in episode_indices
    ]
    return np.concatenate(rows) if rows else np.array([], dtype=np.int64)


def _build_scene_disjoint_split(dataset, train_fraction, seed):
    num_episodes = len(dataset.lengths)
    if num_episodes < 2:
        raise ValueError("Need at least two episodes for a scene-disjoint split.")

    rng = np.random.default_rng(seed)
    episodes = np.arange(num_episodes)
    rng.shuffle(episodes)

    num_val = int(np.ceil(num_episodes * (1.0 - float(train_fraction))))
    num_val = min(max(num_val, 1), num_episodes - 1)
    return np.sort(episodes[num_val:]), np.sort(episodes[:num_val])


def _load_scene_names(dataset):
    metadata_path = dataset.h5_path.with_name("scene_metadata.jsonl")
    if not metadata_path.exists():
        return [f"episode_{idx:04d}" for idx in range(len(dataset.lengths))]

    scene_names = []
    with metadata_path.open() as f:
        for line in f:
            if line.strip():
                scene_names.append(json.loads(line).get("scene_name"))
    if len(scene_names) != len(dataset.lengths):
        logging.warning(
            "Scene metadata count (%d) does not match episode count (%d); using episode ids.",
            len(scene_names),
            len(dataset.lengths),
        )
        return [f"episode_{idx:04d}" for idx in range(len(dataset.lengths))]
    return scene_names


def _save_split(run_dir, dataset, train_episodes, val_episodes, seed, train_fraction):
    scene_names = _load_scene_names(dataset)
    train_episode_set = set(train_episodes.tolist())
    val_episode_set = set(val_episodes.tolist())
    split = {
        "dataset": str(dataset.h5_path),
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "num_episodes": int(len(dataset.lengths)),
        "num_train_episodes": int(len(train_episodes)),
        "num_val_episodes": int(len(val_episodes)),
        "num_train_clips": int(sum(ep in train_episode_set for ep, _ in dataset.clip_indices)),
        "num_val_clips": int(sum(ep in val_episode_set for ep, _ in dataset.clip_indices)),
        "train_episodes": [
            {"episode_index": int(ep), "scene_name": scene_names[int(ep)]}
            for ep in train_episodes
        ],
        "val_episodes": [
            {"episode_index": int(ep), "scene_name": scene_names[int(ep)]}
            for ep in val_episodes
        ],
    }
    split_path = run_dir / f"{dataset.h5_path.stem}_scene_split_seed{seed}.json"
    with split_path.open("w") as f:
        json.dump(split, f, indent=2)
    logging.info("Saved scene-disjoint split to %s", split_path)


def _sanitize_run_name(name):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return name or "run"


def _next_numbered_run_dir(base_dir, output_model_name):
    run_name = _sanitize_run_name(output_model_name)
    existing_numbers = []
    if base_dir.exists():
        for path in base_dir.iterdir():
            if not path.is_dir():
                continue
            match = re.match(r"^(\d+)_", path.name)
            if match:
                existing_numbers.append(int(match.group(1)))
    return base_dir / f"{max(existing_numbers, default=0) + 1}_{run_name}"


def _resolve_run_dir(cfg):
    base_dir = Path(os.environ.get("STABLEWM_HOME", Path.cwd() / "outputs")).expanduser()
    if cfg.get("run_subdir"):
        return base_dir / _sanitize_run_name(cfg.run_subdir)
    if cfg.get("auto_number_run", True):
        return _next_numbered_run_dir(base_dir, cfg.output_model_name)
    return base_dir / str(cfg.get("subdir") or "")


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    action_mode = cfg.get("action_mode", "real")
    if action_mode == "zero":
        batch["action"] = torch.zeros_like(batch["action"])
    elif action_mode != "real":
        raise ValueError(f"Unsupported action_mode={action_mode!r}; expected 'real' or 'zero'.")

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    diagnostics_dict = {
        f"{stage}/{k}": v
        for k, v in _latent_diagnostics(emb).items()
    }
    if stage == "validate":
        self.log_dict(losses_dict, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(diagnostics_dict, on_step=False, on_epoch=True, sync_dist=True)
    else:
        self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(diagnostics_dict, on_step=True, on_epoch=True, sync_dist=True)
    return output


def _latent_diagnostics(emb):
    flat = emb.detach().reshape(-1, emb.shape[-1]).float()
    mean = flat.mean(dim=0)
    std = flat.std(dim=0)
    centered = flat - mean
    cov = centered.T @ centered / max(flat.shape[0] - 1, 1)
    offdiag = cov - torch.diag(torch.diag(cov))
    return {
        "latent_mean_norm": mean.norm(),
        "latent_std_mean": std.mean(),
        "latent_std_min": std.min(),
        "latent_std_max": std.max(),
        "latent_cov_offdiag_abs_mean": offdiag.abs().mean(),
    }


def _build_scheduler_config(cfg):
    scheduler = OmegaConf.to_container(cfg.get("scheduler", {}), resolve=True) or {}
    scheduler_type = scheduler.pop("type", "LinearWarmupCosineAnnealingLR")
    interval = scheduler.pop("interval", "epoch")
    scheduler_cfg = {"type": scheduler_type}

    for key, value in scheduler.items():
        if value is not None:
            scheduler_cfg[key] = value

    if interval == "step":
        max_steps = scheduler_cfg.get("max_steps") or cfg.trainer.get("max_steps")
        if max_steps is not None:
            scheduler_cfg["max_steps"] = int(max_steps)
        if scheduler_cfg.get("warmup_steps") is None:
            raise ValueError("scheduler.warmup_steps must be set when scheduler.interval=step")

    return scheduler_cfg, interval


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = _load_training_dataset(cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    train_episodes, val_episodes = _build_scene_disjoint_split(
        dataset, cfg.train_split, cfg.seed
    )
    train_row_indices = _episode_row_indices(dataset, train_episodes)

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col, row_indices=train_row_indices)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set = EpisodeSubset(dataset, train_episodes)
    val_set = EpisodeSubset(dataset, val_episodes)

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    scheduler_cfg, scheduler_interval = _build_scheduler_config(cfg)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": scheduler_cfg,
            "interval": scheduler_interval,
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = _resolve_run_dir(cfg)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    _save_split(run_dir, dataset, train_episodes, val_episodes, cfg.seed, cfg.train_split)
    logging.info(
        "Using scene-disjoint split: %d train scenes (%d clips), %d val scenes (%d clips)",
        len(train_episodes),
        len(train_set),
        len(val_episodes),
        len(val_set),
    )
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1
    )
    best_object_callback = BestModelObjectCallback(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        monitor="validate/pred_loss_epoch",
        mode="min",
    )
    epoch_metrics_callback = EpochMetricsCallback(
        dirpath=run_dir,
        monitor="validate/pred_loss_epoch",
        mode="min",
    )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.setdefault("default_root_dir", str(run_dir))
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[object_dump_callback, best_object_callback, epoch_metrics_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = cfg.get("resume_ckpt_path")
    if ckpt_path in ("", "null"):
        ckpt_path = None
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path).expanduser()

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
    )

    manager()
    return


if __name__ == "__main__":
    run()
