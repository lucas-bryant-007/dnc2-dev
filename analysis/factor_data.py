"""Dataset dispatch so the factor-geometry drivers run on dsprites OR shapes3d.

Both cores expose the same interface (Cfg, build_arrays, make_eval_loader,
make_paired_loader, FACTOR_COL, ...), so a driver only needs the right core module,
a constructed config, and a display-name map for the factors.
"""
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_loader import namespace_to_dict

# (axis label, (low-half name, high-half name)) per factor, per dataset.
_DISPLAY = {
    "dsprites": {
        "scale": ("size", ("small", "large")),
        "posX": ("x-position", ("left", "right")),
        "posY": ("y-position", ("top", "bottom")),
        "shape": ("shape", ("square", "ellipse")),
        "orientation": ("orientation", ("low", "high")),
    },
    "shapes3d": {
        "object_hue": ("object color", ("cool", "warm")),
        "shape": ("shape", ("angular", "round")),
        "scale": ("size", ("small", "large")),
        "floor_hue": ("floor color", ("cool", "warm")),
        "wall_hue": ("wall color", ("cool", "warm")),
        "orientation": ("orientation", ("left", "right")),
    },
    "mpi3d": {
        "posX": ("x-position", ("left", "right")),
        "posY": ("y-position", ("low", "high")),
        "obj_size": ("size", ("small", "large")),
        "camera": ("camera", ("low", "high")),
        "obj_shape": ("shape", ("a", "b")),
        "obj_color": ("object color", ("cool", "warm")),
        "bg_color": ("background", ("cool", "warm")),
    },
}


def canonical_name(name: str) -> str:
    name = name.lower()
    return "shapes3d" if name in ("shapes3d", "3dshapes") else name


def build_data(cfg):
    """cfg = parsed config namespace. Returns (core_module, data_cfg, display_map)."""
    name = canonical_name(cfg.data.name)
    d = namespace_to_dict(cfg.data)
    if name == "dsprites":
        from data_utils import dsprites_core as core
        data_cfg = core.DSpritesCfg(**d)
    elif name == "shapes3d":
        from data_utils import shapes3d_core as core
        data_cfg = core.Shapes3DCfg(**d)
    elif name == "mpi3d":
        from data_utils import mpi3d_core as core
        data_cfg = core.Mpi3dCfg(**d)
    else:
        raise ValueError(f"Unknown dataset {cfg.data.name!r} "
                         f"(supported: dsprites, shapes3d)")
    return core, data_cfg, _DISPLAY[name]


def display_map(name: str) -> dict:
    return _DISPLAY[canonical_name(name)]
