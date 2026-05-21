import os
import importlib
import json
import logging
import re
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from hdf5_dataset import HDF5Dataset as LocalHDF5Dataset
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def _get_hdf5_dataset_cls():
    if hasattr(swm.data, "HDF5Dataset"):
        return swm.data.HDF5Dataset

    candidates = [
        "stable_worldmodel.data.hdf5",
        "stable_worldmodel.data.hdf5_dataset",
        "stable_worldmodel.data.dataset",
        "stable_worldmodel.data.datasets",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "HDF5Dataset"):
            return module.HDF5Dataset

    return LocalHDF5Dataset


def _load_training_dataset(dataset_cfg, transform=None):
    dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR") or dataset_cfg.pop("cache_dir", None)
    dataset_path = Path(dataset_name).expanduser()
    if cache_dir is not None and not dataset_path.exists():
        cache_dir = Path(cache_dir).expanduser()
        candidates = [
            cache_dir / dataset_name,
            cache_dir / f"{dataset_name}.h5",
            cache_dir / f"{dataset_name}.hdf5",
        ]
        for candidate in candidates:
            if candidate.exists():
                dataset_path = candidate
                dataset_name = str(candidate)
                break

    if dataset_path.suffix.lower() in {".h5", ".hdf5"} and dataset_path.exists():
        HDF5Dataset = _get_hdf5_dataset_cls()
        return HDF5Dataset(name=str(dataset_path), transform=transform, **dataset_cfg)

    if hasattr(swm.data, "load_dataset"):
        return swm.data.load_dataset(
            dataset_name,
            transform=transform,
            **dataset_cfg,
        )

    HDF5Dataset = _get_hdf5_dataset_cls()
    if dataset_path.exists():
        return HDF5Dataset(name=str(dataset_path), transform=transform, **dataset_cfg)
    if cache_dir is not None:
        dataset_cfg["cache_dir"] = str(cache_dir)
    return HDF5Dataset(name=dataset_name, transform=transform, **dataset_cfg)


class EpisodeSubset(torch.utils.data.Dataset):
    """View of an episode dataset restricted to a fixed set of episode indices."""

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


def _build_scene_disjoint_split(dataset, train_fraction, seed):
    num_episodes = len(dataset.lengths)
    if num_episodes < 2:
        raise ValueError("Need at least two episodes for a scene-disjoint train/val split.")

    rng = np.random.default_rng(seed)
    episodes = np.arange(num_episodes)
    rng.shuffle(episodes)

    val_fraction = 1.0 - float(train_fraction)
    num_val = int(np.ceil(num_episodes * val_fraction))
    num_val = min(max(num_val, 1), num_episodes - 1)

    val_episodes = np.sort(episodes[:num_val])
    train_episodes = np.sort(episodes[num_val:])
    return train_episodes, val_episodes


def _save_split(run_dir, dataset, train_episodes, val_episodes, seed, train_fraction):
    scene_names = _load_scene_names(dataset)
    split = {
        "dataset": str(dataset.h5_path),
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "num_episodes": int(len(dataset.lengths)),
        "num_train_episodes": int(len(train_episodes)),
        "num_val_episodes": int(len(val_episodes)),
        "num_train_clips": int(sum(ep in set(train_episodes.tolist()) for ep, _ in dataset.clip_indices)),
        "num_val_clips": int(sum(ep in set(val_episodes.tolist()) for ep, _ in dataset.clip_indices)),
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
    return split_path


def _sanitize_run_name(name):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return name or "run"


def _next_numbered_run_dir(base_dir, output_model_name):
    base_dir = Path(base_dir)
    run_name = _sanitize_run_name(output_model_name)
    existing_numbers = []
    if base_dir.exists():
        for path in base_dir.iterdir():
            if not path.is_dir():
                continue
            match = re.match(r"^(\d+)_", path.name)
            if match:
                existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers, default=0) + 1
    return base_dir / f"{next_number}_{run_name}"


def _resolve_run_dir(cfg):
    base_dir = Path(swm.data.utils.get_cache_dir())
    run_subdir = cfg.get("run_subdir")
    if run_subdir:
        return base_dir / _sanitize_run_name(run_subdir)
    if cfg.get("auto_number_run", True):
        return _next_numbered_run_dir(base_dir, cfg.output_model_name)
    run_id = cfg.get("subdir") or ""
    return base_dir / str(run_id)


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

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = _load_training_dataset(cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

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

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set = EpisodeSubset(dataset, train_episodes)
    val_set = EpisodeSubset(dataset, val_episodes)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
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
    split_path = _save_split(run_dir, dataset, train_episodes, val_episodes, cfg.seed, cfg.train_split)
    logging.info(
        "Using scene-disjoint split: %d train scenes (%d clips), %d val scenes (%d clips)",
        len(train_episodes), len(train_set), len(val_episodes), len(val_set)
    )
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.setdefault("default_root_dir", str(run_dir))

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
