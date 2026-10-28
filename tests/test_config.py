"""Test that configs/default.yaml loads and that the derived values are right.

What is checked:
    - the YAML loads into a Config object with the approved values
    - derived values: J = 64 patches, sequence length M = 65, E = 384,
      class token index 32
    - command-line style overrides ("train.epochs=5") change the right value,
      get the right type, and reject typos
    - the config survives a round trip to a plain dictionary (as in checkpoints)
"""

import os  # for building file paths
import sys  # for editing the module search path

# Add the project root to Python's search path so `import vim` works
# even when this file is run directly (python tests/test_config.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # test framework; used here for `pytest.raises`

from vim.config import config_from_dict, config_to_dict, load_config  # under test

# Absolute path of the default config, so the test works from any current folder.
# os.path.dirname(__file__) is the tests/ folder; ".." steps up to the repo root.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def test_default_config_loads_with_approved_values():
    # Read the real YAML file
    cfg = load_config(CONFIG_PATH)

    # The supervisor-approved scaled-down configuration
    assert cfg.model.img_size == 128
    assert cfg.model.patch_size == 16
    assert cfg.model.embed_dim == 192
    assert cfg.model.d_state == 16
    assert cfg.model.depth == 12
    assert cfg.model.num_classes == 5
    assert cfg.model.cls_position == "middle"
    assert cfg.model.conv_kernel == 4

    # The paper's optimiser values (Section 4.1)
    assert cfg.train.lr == pytest.approx(1e-3)
    assert cfg.train.weight_decay == pytest.approx(0.05)

    # The seed is an integer, as set_seed expects
    assert isinstance(cfg.seed, int)


def test_derived_values():
    # Same config, now checking the values computed by the @property methods
    cfg = load_config(CONFIG_PATH)

    # Print them so they are visible with `pytest -s` or when run directly
    print("grid_size  :", cfg.model.grid_size)
    print("num_patches:", cfg.model.num_patches)
    print("seq_len    :", cfg.model.seq_len)
    print("expand_dim :", cfg.model.expand_dim)
    print("cls_index  :", cfg.model.cls_index)

    # 128 / 16 = 8 patches per side
    assert cfg.model.grid_size == 8
    # J = 8 * 8 = 64 patches
    assert cfg.model.num_patches == 64
    # M = 64 patches + 1 class token = 65
    assert cfg.model.seq_len == 65
    # E = 2 * 192 = 384
    assert cfg.model.expand_dim == 384
    # Middle class token: 64 // 2 = 32
    assert cfg.model.cls_index == 32


def test_overrides_change_value_and_type():
    # Two overrides, as they would arrive from the command line
    overrides = ["train.epochs=5", "model.cls_position=head"]
    cfg = load_config(CONFIG_PATH, overrides=overrides)

    # The integer arrived as an int, not as the string "5"
    assert cfg.train.epochs == 5
    assert isinstance(cfg.train.epochs, int)

    # Changing the class token position changes the derived index to the front
    assert cfg.model.cls_index == 0


def test_override_typo_is_rejected():
    # A misspelled key must fail loudly, not be silently ignored.
    # `with pytest.raises(KeyError):` passes only if the block raises a KeyError.
    with pytest.raises(KeyError):
        load_config(CONFIG_PATH, overrides=["train.epochz=5"])


def test_round_trip_through_dict():
    # Config -> plain dict -> Config should give back an equal object.
    # (Dataclasses get an automatic __eq__ that compares every field.)
    cfg = load_config(CONFIG_PATH)
    again = config_from_dict(config_to_dict(cfg))
    assert again == cfg


# Allow running as a plain script: python tests/test_config.py
if __name__ == "__main__":
    test_default_config_loads_with_approved_values()
    test_derived_values()
    test_overrides_change_value_and_type()
    test_override_typo_is_rejected()
    test_round_trip_through_dict()
    print("OK")
