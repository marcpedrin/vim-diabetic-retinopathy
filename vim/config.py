"""Config: reads configs/default.yaml into typed Python objects (dataclasses).

What this file does:
    Loads the YAML file of hyperparameters and turns it into a `Config` object,
    so the rest of the code can write `cfg.model.embed_dim` instead of hunting
    for a magic number. It also computes the values that FOLLOW from the
    settings (number of patches, sequence length, expanded dimension E, the
    index of the class token) so they are computed in exactly one place.

Vim paper correspondence (arXiv 2401.09417):
    No equation is implemented here. The settings themselves come from the
    paper's Vim-Ti model (Section 4.1 and Table 1 of the paper): D = 192,
    E = 384, N = 16, patch size 16. The derived values match the symbols in
    Section 3.2: J = number of patches, and the sequence length is J + 1
    (the patches plus one class token).

Input / output:
    load_config("configs/default.yaml") -> Config
    No tensors are involved in this file.

Deliberate differences from the paper:
    None in this file. The scaled-down values (128-pixel images, 12 blocks)
    are documented next to each value in configs/default.yaml.

Why a config object beats global variables:
    In Java you might put constants in a `static final` field of some class.
    That is fine for true constants, but hyperparameters CHANGE between runs.
    If they were globals, any function could silently read or modify them,
    and a test could not use a smaller model without editing the file. With a
    config object we build it once, pass it explicitly to whoever needs it,
    and a test can make its own modified copy without affecting anyone else.
"""

# `import x` loads a module (Python's word for a file of code, like a Java class
# file you can import). `from x import y` pulls just the name `y` out of module `x`.
import copy  # standard library: `copy.deepcopy` makes a fully independent copy
from dataclasses import asdict, dataclass, field  # standard library, explained below

import yaml  # PyYAML: reads .yaml files into ordinary Python dictionaries


# -------------------------------------------------------------------------------------
# PYTHON CONCEPT: classes, `self`, decorators and dataclasses
#
# A Python `class` is like a Java class. Methods are written with `def` inside it.
# Python has no hidden `this`: every method takes the object explicitly as its first
# parameter, named `self` by convention. So `cfg.num_patches` inside the class is
# written `self.num_patches`, exactly where Java would write `this.numPatches`.
#
# A line starting with `@` just above a class or function is a DECORATOR. It is a
# function that receives the class/function you just wrote and returns an improved
# version of it. Java's nearest relative is an annotation like `@Override`, but a
# Python decorator really runs code and can change what you defined.
#
# `@dataclass` is a decorator from the standard library. You list the fields with
# their types and default values, and it writes the boring code for you: a
# constructor `__init__` that takes every field as an argument, a readable
# `__repr__` (Java's toString) and an `__eq__` (Java's equals). It is roughly what a
# Java `record` gives you. `name: int = 128` means "field `name`, of type int,
# default 128". Python does not ENFORCE these types at run time; they are
# documentation for humans and for editor tools.
# -------------------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Shape of the Vision Mamba network. Each field mirrors `model:` in the YAML."""

    # Input image side, pixels (paper: 224)
    img_size: int = 128
    # Patch side, pixels (paper: 16)
    patch_size: int = 16
    # Colour channels (RGB)
    in_chans: int = 3
    # D: embedding dimension (paper Vim-Ti: 192)
    embed_dim: int = 192
    # E / D: expansion factor inside each block (paper: 2)
    expand: int = 2
    # N: SSM hidden state size per channel (paper: 16)
    d_state: int = 16
    # L: number of stacked Vim blocks (paper Vim-Ti: 24)
    depth: int = 12
    # Output classes: DR grades 0-4
    num_classes: int = 5
    # "middle" or "head": where the class token goes (paper Table 5)
    cls_position: str = "middle"
    # Kernel size of the depthwise conv1d in each block (Algorithm 1, line 7)
    conv_kernel: int = 4
    # Range for the initial step size Delta (Mamba's initialisation)
    dt_min: float = 0.001
    dt_max: float = 0.1

    # PYTHON CONCEPT: `@property`. This decorator turns a method into something you
    # read like a field: you write `cfg.num_patches`, with no brackets, and Python
    # quietly calls this method. It is a Java getter without the `get...()` noise.
    # We use it for values DERIVED from other fields, so they can never get out of
    # sync with them (there is nothing stored that could become stale).
    @property
    def grid_size(self) -> int:
        # Patches along one side of the image: 128 // 16 = 8.
        # `//` is integer division in Python (like `/` between two ints in Java);
        # a single `/` in Python always gives a float, e.g. 7 / 2 = 3.5.
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        # J in the paper (Section 3.2): total patches = 8 * 8 = 64
        return self.grid_size * self.grid_size

    @property
    def seq_len(self) -> int:
        # M in Algorithm 1: the sequence length the blocks see, J patches + 1 class
        # token = 64 + 1 = 65
        return self.num_patches + 1

    @property
    def expand_dim(self) -> int:
        # E in Algorithm 1: the widened channel count inside a block, 2 * 192 = 384
        return self.expand * self.embed_dim

    @property
    def cls_index(self) -> int:
        # The position the class token occupies in the 65-long sequence.
        # "middle": 64 // 2 = 32, so 32 patches sit before it and 32 after it.
        if self.cls_position == "middle":
            return self.num_patches // 2
        # "head": the very front, as in ViT (paper Table 5's "head class token").
        if self.cls_position == "head":
            return 0
        # Anything else is a typo in the YAML; fail loudly instead of guessing.
        # PYTHON CONCEPT: f-strings. A string with an `f` in front can contain
        # `{expression}` placeholders that are filled in with the value, like
        # Java's String.format but written inline. `!r` shows the value with quotes.
        raise ValueError(
            f"cls_position must be 'middle' or 'head', got {self.cls_position!r}"
        )


@dataclass
class DataConfig:
    """Dataset locations and preprocessing. Mirrors `data:` in the YAML."""

    # EyePACS (training) image folder, label CSV and file extension
    eyepacs_image_dir: str = "datasets/eyepacs/train"
    eyepacs_labels_csv: str = "datasets/eyepacs/trainLabels.csv"
    eyepacs_image_ext: str = ".jpeg"
    # Messidor-2 (evaluation only) image folder, label CSV and column names
    messidor_image_dir: str = "datasets/messidor2/images"
    messidor_labels_csv: str = "datasets/messidor2/messidor_data.csv"
    messidor_id_column: str = "image_id"
    messidor_label_column: str = "adjudicated_dr_grade"
    messidor_image_ext: str = ".png"
    # Fractions of patients held out for validation and test
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    # Brightness at or below which a pixel counts as black border
    crop_threshold: int = 10
    # Augmentation strengths (training only)
    rotation_degrees: float = 15.0
    color_jitter: float = 0.1
    # DataLoader settings
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """Optimisation recipe. Mirrors `train:` in the YAML."""

    # Passes over the training set (paper: 300)
    epochs: int = 30
    # Images per step (paper: 1024)
    batch_size: int = 32
    # AdamW learning rate and weight decay (paper: 1e-3 and 0.05)
    lr: float = 1.0e-3
    weight_decay: float = 0.05
    # Linear warmup length and cosine floor
    warmup_epochs: int = 3
    min_lr: float = 1.0e-5
    # Maximum gradient norm
    grad_clip: float = 1.0
    # Mixed precision on GPU (speed only)
    amp: bool = True
    # Inverse-frequency class weights in the loss
    class_weights: bool = True
    # Folder for checkpoints and logs
    output_dir: str = "outputs/default"
    # Settings for the --smoke-test quick check
    smoke_test_images: int = 100
    smoke_test_epochs: int = 2
    smoke_test_batch_size: int = 8


@dataclass
class EvalConfig:
    """Evaluation settings. Mirrors `eval:` in the YAML."""

    # Checkpoint to load for evaluation
    checkpoint: str = "outputs/default/best.pt"
    # Where the metrics JSON is written
    results_json: str = "outputs/default/results.json"
    # Where the report figures are saved
    figures_dir: str = "report/figures"


@dataclass
class Config:
    """The whole experiment: one sub-config per YAML section, plus the seed."""

    # Master random seed (see train/utils.py set_seed)
    seed: int = 42
    # `field(default_factory=ModelConfig)` means "when a Config is created without a
    # model argument, call ModelConfig() to make a FRESH one". A plain
    # `= ModelConfig()` would make every Config share ONE ModelConfig object, so
    # changing one config would secretly change all of them; dataclasses forbid it.
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def apply_override(raw: dict, override: str) -> None:
    """Change one value in the raw YAML dictionary, e.g. "train.epochs=5".

    This lets a command line say `--override train.epochs=5` without editing the
    YAML file. `raw` is modified in place (nothing is returned).
    """
    # The override must look like "section.key=value"; reject anything else early.
    if "=" not in override:
        raise ValueError(
            f"Override must look like 'section.key=value', got {override!r}"
        )

    # Split only at the FIRST "=", so values that contain "=" survive intact.
    # "train.epochs=5" -> dotted_key = "train.epochs", value_text = "5"
    dotted_key, value_text = override.split("=", 1)

    # "train.epochs" -> ["train", "epochs"]. A top-level key such as "seed" gives
    # a list of length one: ["seed"].
    parts = dotted_key.split(".")

    # Walk down the nested dictionaries to the one that holds the final key.
    # For ["train", "epochs"] this moves `node` from the whole config to raw["train"].
    # `parts[:-1]` is SLICING: every element except the last one.
    node = raw
    for part in parts[:-1]:
        # Refuse to create new sections; a typo like "trian.epochs" should fail.
        if part not in node:
            raise KeyError(f"Unknown config section {part!r} in override {override!r}")
        node = node[part]

    # `parts[-1]` is the last element ("epochs"); negative indices count from the end.
    key = parts[-1]
    # Same protection for the final key.
    if key not in node:
        raise KeyError(f"Unknown config key {key!r} in override {override!r}")

    # Let YAML decide the type, exactly as if the value had been written in the file:
    # "5" -> int 5, "1e-4" -> float, "true" -> bool True, "abc" -> str "abc".
    node[key] = yaml.safe_load(value_text)


def config_from_dict(raw: dict) -> Config:
    """Build a Config from a plain nested dictionary (as read from YAML)."""
    # PYTHON CONCEPT: `**dictionary` (keyword-argument unpacking). Writing
    # `ModelConfig(**{"img_size": 128, "depth": 12})` is exactly the same as writing
    # `ModelConfig(img_size=128, depth=12)`: every key becomes a named argument.
    # A bonus: if the YAML contains a key the dataclass does not have (a typo such
    # as "embed_dimm"), Python raises a TypeError instead of silently ignoring it.
    # `raw.get("model", {})` returns raw["model"], or an empty dict if it is missing,
    # in which case every field keeps its default.
    return Config(
        seed=raw.get("seed", 42),
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )


def load_config(path: str, overrides=None) -> Config:
    """Read a YAML file, apply any "section.key=value" overrides, return a Config.

    Args:
        path: path to a YAML file such as "configs/default.yaml".
        overrides: optional list of strings like ["train.epochs=5"].
    """
    # PYTHON CONCEPT: the `with` block. `open(...)` gives a file handle that must be
    # closed afterwards. `with open(...) as f:` closes it automatically when the
    # indented block ends, even if an error happens inside. It is Java's
    # try-with-resources: `try (FileReader f = new FileReader(path)) { ... }`.
    # `encoding="utf-8"` avoids Windows guessing a different text encoding.
    with open(path, "r", encoding="utf-8") as f:
        # safe_load turns the YAML text into nested dicts/lists/numbers/strings.
        # ("safe" = it refuses YAML tags that could construct arbitrary objects.)
        raw = yaml.safe_load(f)

    # `None` is Python's null. `or []` means "if overrides is None (or empty), use an
    # empty list instead", so the loop below simply does nothing.
    for override in overrides or []:
        apply_override(raw, override)

    # Turn the plain dictionaries into typed dataclasses.
    return config_from_dict(raw)


def config_to_dict(cfg: Config) -> dict:
    """Turn a Config back into plain nested dictionaries (for saving or printing).

    Checkpoints store this dictionary so a saved model always carries the exact
    settings that built it.
    """
    # `asdict` (from dataclasses) converts a dataclass, and every dataclass inside
    # it, into ordinary dicts. Properties such as num_patches are NOT included,
    # because they are computed, not stored. deepcopy guarantees the caller gets a
    # copy that is fully independent of `cfg`.
    return copy.deepcopy(asdict(cfg))
