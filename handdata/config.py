"""YAML 설정 로드. 속성 접근(cfg.lifting.max_reproj_px) + 상대경로는 프로젝트 루트 기준으로 해석."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def resolve(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else Path(self["_root"]) / p

    def to_plain(self) -> dict:
        d = copy.deepcopy(dict(self))
        d.pop("_root", None)
        return _unwrap(d)


def _wrap(x):
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _unwrap(x):
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unwrap(v) for v in x]
    return x


def load_config(path="configs/default.yaml", overrides=None) -> Cfg:
    """overrides: ["lifting.finger_method=ray_bone", "board.square_length=0.034"] (ablation용)."""
    path = Path(path).resolve()
    cfg = _wrap(yaml.safe_load(path.read_text()))
    cfg["_root"] = str(path.parent.parent)   # configs/ 의 상위 = 프로젝트 루트
    for item in overrides or []:
        key, _, val = item.partition("=")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for k in parents:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"없는 설정 키: {key}")
        node[leaf] = _wrap(yaml.safe_load(val))
    return cfg


def add_config_args(p) -> None:
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="설정 덮어쓰기, 예: --set lifting.finger_method=ray_bone hand_model.plane_offset=0.008")
