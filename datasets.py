"""
datasets.py

Unified dataset loader for WiFi HAR Benchmark

Supported datasets
------------------
1. UT_HAR
2. SSHAR_ESP
3. SSHAR_asus
4. XRF55
"""

from pathlib import Path
import os
import glob
import re

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Default split mode when none is passed: "repeat" (by repetition index),
# "user" (cross-user, train users given by USER_CASES[...][case_split]) or
# "position" (SSHAR only, cross-position: test = positions with
# pos % N_POSITION_CASES == case_split, i.e. 2 of the 8 positions per case).
SPLIT_MODES = ("repeat", "user", "position")
N_POSITION_CASES = 4
DEFAULT_SPLIT_MODE = "repeat"


def _check_split_mode(split_mode):
    split_mode = (split_mode or DEFAULT_SPLIT_MODE).lower()
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got '{split_mode}'")
    return split_mode


# ---- SSHAR: fixed train/val/test rules of SSHAR_ESP_protocol.md (7:1:2) ----
# Every model in the benchmark uses these rules -> 4480 / 640 / 1280 samples.
SSHAR_VAL_SUBJECTS = [12, 13, 1, 2, 10]   # val subject of user case 0..4
SSHAR_ALL_SUBJECTS = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14]


def sshar_repeat_split_of(pos, direction, rep):
    """Protocol 3B: per (subject, action) 80 recordings -> 56 / 8 / 16."""
    if direction == 0:                      # dir00: 10 repetitions
        return "test" if rep > 8 else "val" if rep == 8 else "train"
    if rep == 5:                            # dir01 / dir02: 5 repetitions
        return "test"
    val_dir = 1 if pos % 2 else 2           # odd pos -> dir01, even pos -> dir02
    return "val" if rep == 4 and direction == val_dir else "train"


def sshar_user_split_of(subject_id, user_case=0):
    """Protocol 3A: 7 train / 1 val / 2 test subjects."""
    if subject_id not in USER_CASES["sshar"][user_case]:
        return "test"
    return "val" if subject_id == SSHAR_VAL_SUBJECTS[user_case] else "train"


# ---- Other splits (XRF55, SSHAR position) ----
#   repeat           -> all samples split 7:1:1, stratified by class (XRF55).
#   user / position  -> test = held-out users/positions;
#                       the rest is split train:val = 7:1, stratified by class.
# The split uses its own fixed seed, so it is identical across runs/models.
SPLIT_RATIO = (7, 1, 1)
SPLIT_SEED = 42


def _assign_splits(labels, held_out, carve_test, ratio=SPLIT_RATIO, seed=SPLIT_SEED):
    """Return an array of 'train' / 'val' / 'test', one per sample.

    labels   : class index per sample (samples must be in a deterministic order)
    held_out : True if the sample is already test (user / position split)
    carve_test : True for 'repeat' (test is a random 1/9); False for user /
                 position (test = held_out only, rest split train:val)
    """
    labels = np.asarray(labels)
    held_out = np.asarray(held_out, dtype=bool)
    out = np.full(len(labels), "train", dtype=object)
    out[held_out] = "test"

    n_train, n_val, n_test = ratio
    if not carve_test:
        n_test = 0  # test already fixed, only carve val out of the rest
    total = n_train + n_val + n_test

    rng = np.random.RandomState(seed)
    for c in np.unique(labels):
        idx = np.where((labels == c) & ~held_out)[0]
        rng.shuffle(idx)
        k_test = int(round(len(idx) * n_test / total))
        k_val = int(round(len(idx) * n_val / total))
        out[idx[:k_test]] = "test"
        out[idx[k_test:k_test + k_val]] = "val"
    return out

# ============================================================
# Default configuration
# ============================================================

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_PIN_MEMORY = True

# Keep only the requested classes in the benchmark.
# Labels are remapped to contiguous indices 0..5 for PyTorch.
DEFAULT_SELECTED_LABELS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_LABEL_MAP = {
    label: idx for idx, label in enumerate(DEFAULT_SELECTED_LABELS)
}

INTEL_30_SUBCARRIERS = [
    -28, -26, -24, -22, -20, -18, -16, -14,
    -12, -10, -8, -6, -4, -2,
    -1, 1, 3, 5, 7, 9, 11, 13,
    15, 17, 19, 21, 23, 25, 27, 28
]

# Map signed subcarrier -> index 0..55
INTEL_30_INDICES = [
    sc + 28 if sc < 0 else sc + 27
    for sc in INTEL_30_SUBCARRIERS
]

USER_CASES = {
    "xrf55": [
        list(range(1, 25)),
        [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23, 24, 25, 27, 28, 29, 30],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29],
        [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 29, 30]
    ],
    "sshar": [
        [1, 2, 3, 4, 9, 10, 11, 12],
        [1, 3, 4, 9, 11, 12, 13, 14],
        [1, 2, 4, 9, 10, 12, 13, 14],
        [2, 3, 4, 10, 11, 12, 13, 14],
        [1, 2, 4, 9, 10, 11, 13, 14],
    ]
}
# ============================================================
# Utility
# ============================================================

def scan_folders(root: Path, prefix: str):
    """
    Scan folders with given prefix.

    Example
    -------
    scan_folders(data_root, "room")
        -> ["room_01","room_02"]

    scan_folders(room_path, "subject")
        -> ["subject_01",...]
    """

    if not root.exists():
        return []

    folders = []

    for item in sorted(root.iterdir()):

        if item.is_dir() and item.name.startswith(prefix):
            folders.append(item.name)

    return folders


# ============================================================
#                    UT_HAR
# ============================================================

class UTHARDataset(Dataset):
    """
    UT_HAR Dataset

    Folder structure

    UT_HAR
    ├── data
    │     train_data.csv
    │     val_data.csv
    │     test_data.csv
    │
    └── label
          train_label.csv
          val_label.csv
          test_label.csv

    Parameters
    ----------
    root_dir

    split
        train
        val
        test

    normalize

    norm_type
        minmax
        zscore

    mean,std
        only used for zscore
    """

    def __init__(
        self,
        root_dir,
        split="train",
        normalize=True,
        norm_type="minmax",
        mean=None,
        std=None,
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.normalize = normalize
        self.norm_type = norm_type
        self.mean = mean
        self.std = std

        self.X = None
        self.y = None

        self.load_data()

    # --------------------------------------------------------

    def _find_ut_har_dir(self):
        """
        Locate the folder that contains data/X_{split}.csv and label/y_{split}.csv.
        Accepts root_dir pointing at the parent of UT_HAR, at UT_HAR itself,
        or anywhere above it (e.g. a Kaggle input dir with nested folders).
        """
        fname = f"X_{self.split}.csv"
        for cand in (self.root_dir / "UT_HAR", self.root_dir):
            if (cand / "data" / fname).exists():
                return cand

        if self.root_dir.exists():
            for hit in sorted(self.root_dir.rglob(fname)):
                if hit.parent.name == "data":
                    return hit.parent.parent

        raise FileNotFoundError(
            f"Could not find data/{fname} under '{self.root_dir}'. "
            f"Check --data_root (it should point at the UT_HAR folder or one of its parents)."
        )

    def load_data(self):
        base = self._find_ut_har_dir()
        data_file = base / "data" / f"X_{self.split}.csv"
        label_file = base / "label" / f"y_{self.split}.csv"

        if not label_file.exists():
            raise FileNotFoundError(label_file)

        with open(data_file, "rb") as f:
            x = np.load(f)

        # Raw data shape: (N, 250, 90) = (N, time, subcarrier)
        # Transpose to (N, 90, 250) = (N, subcarrier, time)
        # to match model input format (C, T)
        x = x.transpose(0, 2, 1)

        with open(label_file, "rb") as f:
            y = np.load(f)

        # one-hot -> class index
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        self.X = x.astype(np.float32)
        self.y = y.astype(np.int64)

    # --------------------------------------------------------
    def __len__(self):
        return len(self.X)
    
    # --------------------------------------------------------
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.normalize:
            if self.norm_type == "minmax":
                xmin = x.min()
                xmax = x.max()
                x = (
                    x - xmin
                ) / (
                    xmax - xmin + 1e-8
                )
            elif self.norm_type == "zscore":
                if self.mean is None or self.std is None:
                    raise ValueError(
                        "mean/std required for zscore"
                    )
                x = (
                    x - self.mean
                ) / (
                    self.std + 1e-8
                )
            else:
                raise ValueError(
                    f"Unknown norm_type {self.norm_type}"
                )

        x = torch.from_numpy(x).float()
        y = int(self.y[idx])

        return x, y


# ============================================================
# compute mean/std for UT_HAR
# ============================================================
def compute_ut_har_mean_std(root_dir):
    """
    Compute global mean/std
    using TRAIN split only.

    Only used when norm_type='zscore'
    """

    dataset = UTHARDataset(
        root_dir=root_dir,
        split="train",
        normalize=False,
    )

    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    n = len(dataset)
    print(f"[XRF55] computing mean/std over {n} train samples ...")
    for i, (x, _) in enumerate(dataset):
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"[XRF55]   mean/std {i + 1}/{n}")
        x = x.numpy().astype(np.float64)
        total_sum += x.sum()
        total_sq += np.square(x).sum()
        total_count += x.size

    mean = total_sum / total_count

    std = np.sqrt(
        total_sq / total_count - mean ** 2
    )

    print(f"UT_HAR mean = {mean:.6f}")
    print(f"UT_HAR std  = {std:.6f}")

    return mean, std


# ============================================================
#               Base SSHAR Dataset
# ============================================================
class _SSHARDatasetBase(Dataset):
    """
    Internal SSHAR dataset.

    Do not use directly.

    Use

        SSHAR_ESP_Dataset

    or

        SSHAR_ASUS_Dataset
    """

    pattern = re.compile(
        r"act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)"
    )
    # --------------------------------------------------------

    def __init__(
        self,
        root_dir,
        device,
        signal="amp",
        split="train",
        case_split = 0,
        split_mode=DEFAULT_SPLIT_MODE,
        shape_option="2d",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        target_time=None,
        rooms=None,
        subjects=None,
        rx_list=None,
        intel_shape=False,
        allowed_labels=None,
    ):

        self.root_dir = Path(root_dir)
        self.device = device
        self.signal = signal
        self.split = split
        self.case_split = case_split
        self.split_mode = _check_split_mode(split_mode)
        self.shape_option = shape_option
        self.normalize = normalize
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.target_time = target_time
        self.intel_shape = intel_shape
        self.allowed_labels = tuple(DEFAULT_SELECTED_LABELS) if allowed_labels is None else tuple(allowed_labels)
        self.label_map = {label: idx for idx, label in enumerate(self.allowed_labels)}
        # ----------------------------------------
        if rooms is None:
            rooms = scan_folders(
                self.root_dir,
                "room",
            )

        self.rooms = rooms

        # ----------------------------------------
        if rx_list is None:
            rx_list = [
                "rx_00",
                "rx_01",
                "rx_02",
            ]
        self.rx_list = rx_list

        # ----------------------------------------
        if subjects is None:
            subjects = []
            for room in self.rooms:
                base = (
                    self.root_dir /
                    room /
                    self.device /
                    self.rx_list[0]
                )

                subjects.extend(
                    scan_folders(
                        base,
                        "subject",
                    )
                )

            subjects = sorted(
                list(
                    set(subjects)
                )
            )

        self.subjects = subjects

        # ----------------------------------------

        self.samples = []
        self.build_index()

    # --------------------------------------------------------
    def build_index(self):

        """
        Build sample index.

        Only scan RX00.

        Store all RX file paths.

        samples[i] =

        {
            "label":0,
            "files":[
                rx00,
                rx01,
                rx02
            ]
        }
        """
        self.samples.clear()
        for room in self.rooms:
            for subject in self.subjects:
                folder = (
                    self.root_dir /
                    room /
                    self.device /
                    self.rx_list[0] /
                    subject
                )

                if not folder.exists():
                    continue

                for file in sorted(
                    folder.iterdir()
                ):
                    if not file.name.startswith(
                        self.signal
                    ):
                        continue

                    m = self.pattern.search(
                        file.name
                    )
                    if m is None:
                        continue

                    act = int(m.group(1))
                    pos = int(m.group(2))
                    direction = int(m.group(3))
                    rep = int(m.group(4))

                    if act not in self.label_map:
                        continue

                    
                    fixed = None   # split given directly by the protocol
                    held_out = False
                    if self.split_mode == "repeat":
                        # Protocol 3B: theo lan lap, 7:1:2
                        fixed = sshar_repeat_split_of(pos, direction, rep)
                    elif self.split_mode == "position":
                        # Chia theo vi tri: giu lai 2/8 vi tri lam test, val = 1/8 phan con lai
                        held_out = pos % N_POSITION_CASES == self.case_split
                    else:
                        # Protocol 3A: theo nguoi, 7 / 1 / 2 nguoi
                        fixed = sshar_user_split_of(int(subject.split("_")[1]), self.case_split)

                    rx_files = []

                    for rx in self.rx_list:
                        rx_files.append(
                            self.root_dir
                            / room
                            / self.device
                            / rx
                            / subject
                            / file.name
                        )

                    self.samples.append(
                        {
                            "label": self.label_map[act],
                            "room": room,
                            "subject": subject,
                            "pos": pos,
                            "dir": direction,
                            "rep": rep,
                            "files": rx_files,
                            "held_out": held_out,
                            "fixed": fixed,
                        }
                    )

        if self.split_mode == "position":
            splits = _assign_splits(
                [s["label"] for s in self.samples],
                [s["held_out"] for s in self.samples],
                carve_test=False,
            )
        else:
            splits = [s["fixed"] for s in self.samples]
        self.samples = [s for s, sp in zip(self.samples, splits) if sp == self.split]

    # --------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]
        rx_data = []

        for file in sample["files"]:
            x = np.load(file).astype(np.float32)
            rx_data.append(x)

        # --------------------------------------------------
        # Stack RX
        #
        # ESP:
        #   (3, 1, 56, T)
        #
        # ASUS:
        #   (3, 4, 56, T)
        # --------------------------------------------------
        x = np.stack(rx_data)

        # ============================================================
        # Intel shape
        # ============================================================
        if self.intel_shape:
            # Select antennas
            if self.device == "esp":
                # ESP32: 1 antenna
                antenna_indices = [0]
            elif self.device == "asus":
                # ASUS: keep antenna 0, 1, 3
                antenna_indices = [0,1,2,3]
            else:
                raise ValueError(
                    f"Unknown device: {self.device}"
                )
            x = x[:, antenna_indices, :, :]
            # Select 30 Intel-like subcarriers
            # x = x[:, :, INTEL_30_INDICES, :]

        # --------------------------------------------------
        # Shape
        # --------------------------------------------------
        if self.shape_option == "2d":
            # (rx, antenna, sub, time)
            # -> (rx * antenna * sub, time)
            x = x.reshape(
                -1,
                x.shape[-1]
            )
        elif self.shape_option == "3d":
            # (rx, antenna, sub, time)
            # -> (rx * antenna, sub, time)
            rx, antenna, sub, time_len = x.shape
            x = x.reshape(
                rx * antenna,
                sub,
                time_len
            )
        else:
            raise ValueError(
                f"Unknown shape_option: {self.shape_option}"
            )

        # --------------------------------------------------
        # Normalization
        # --------------------------------------------------
        if self.normalize:
            if self.norm_type == "zscore":
                if self.mean is None or self.std is None:
                    raise ValueError(
                        "mean/std required for zscore"
                    )
                x = (
                    x - self.mean
                ) / (
                    self.std + 1e-8
                )
            elif self.norm_type == "minmax":
                xmin = x.min()
                xmax = x.max()
                x = (
                    x - xmin
                ) / (
                    xmax - xmin + 1e-8
                )
            else:
                raise ValueError(
                    f"Unknown norm_type {self.norm_type}"
                )

        x = torch.from_numpy(x).float()

        if self.target_time is not None and x.ndim in (2, 3) and x.shape[-1] != self.target_time:
            x = F.adaptive_max_pool1d(x, output_size=self.target_time)

        y = sample["label"]
        return x, y


# ============================================================
# SSHAR ESP
# ============================================================

class SSHAR_ESP_Dataset(_SSHARDatasetBase):
    def __init__(
        self,
        root_dir,
        signal="amp",
        split="train",
        case_split = 0,
        split_mode=DEFAULT_SPLIT_MODE,
        shape_option="2d",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        target_time=None,
        rooms=None,
        subjects=None,
        rx_list=None,
        intel_shape=False,
        allowed_labels=None,
    ):
        super().__init__(
            root_dir=root_dir,
            device="esp",
            signal=signal,
            split=split,
            case_split = case_split,
            split_mode=split_mode,
            shape_option=shape_option,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            target_time=target_time,
            rooms=rooms,
            subjects=subjects,
            rx_list=rx_list,
            intel_shape=intel_shape,
            allowed_labels=allowed_labels,
        )

# ============================================================
# SSHAR asus
# ============================================================
class SSHAR_ASUS_Dataset(_SSHARDatasetBase):
    def __init__(
        self,
        root_dir,
        signal="amp",
        split="train",
        case_split = 0,
        split_mode=DEFAULT_SPLIT_MODE,
        shape_option="2d",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        target_time=None,
        rooms=None,
        subjects=None,
        rx_list=None,
        intel_shape=False,
        allowed_labels=None,
    ):
        super().__init__(
            root_dir=root_dir,
            device="asus",
            signal=signal,
            split=split,
            case_split = case_split,
            split_mode=split_mode,
            shape_option=shape_option,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            target_time=target_time,
            rooms=rooms,
            subjects=subjects,
            rx_list=rx_list,
            intel_shape=intel_shape,
            allowed_labels=allowed_labels,
        )

# ============================================================
# Compute SSHAR mean/std
# ============================================================
def compute_sshar_mean_std(dataset):
    """
    Parameters
    ----------
    dataset

        SSHAR_ESP_Dataset
        or
        SSHAR_ASUS_Dataset

        normalize=False
    """

    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    n = len(dataset)
    print(f"[XRF55] computing mean/std over {n} train samples ...")
    for i, (x, _) in enumerate(dataset):
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"[XRF55]   mean/std {i + 1}/{n}")
        x = x.numpy().astype(np.float64)
        total_sum += x.sum()
        total_sq += np.square(x).sum()
        total_count += x.size

    mean = total_sum / total_count
    std = np.sqrt(
        total_sq / total_count - mean ** 2
    )

    print(f"SSHAR mean = {mean:.6f}")
    print(f"SSHAR std  = {std:.6f}")

    return mean, std


# ============================================================
# XRF55 Dataset (Subset)
# ============================================================
_XRF55_FILE_CACHE = {}


def find_xrf55_wifi_files(root_dir):
    """
    Locate XRF55 WiFi .npy files.

    1. *.npy directly inside root_dir  (root_dir is the WiFi folder)
    2. otherwise *.npy inside sub-folders named 'WiFi' (case-insensitive),
       e.g. root/train_data/WiFi, root/test_data/WiFi.
       Other modalities (mmWave, RFID, ...) are never included.
    """
    root_dir = Path(root_dir)
    key = str(root_dir.resolve())
    if key in _XRF55_FILE_CACHE:
        return _XRF55_FILE_CACHE[key]

    files = sorted(root_dir.glob("*.npy"))
    if files:
        print(f"[XRF55] {len(files)} .npy files in {root_dir}")
    else:
        wifi_dirs = sorted(
            d for d in root_dir.rglob("*")
            if d.is_dir() and d.name.lower() == "wifi"
        )
        for d in wifi_dirs:
            found = sorted(d.glob("*.npy"))
            print(f"[XRF55] {len(found)} .npy files in {d}")
            files.extend(found)
        if not wifi_dirs:
            print(f"[XRF55] no .npy files in {root_dir} and no 'WiFi' sub-folder found")

    _XRF55_FILE_CACHE[key] = files
    return files


class XRF55Dataset(Dataset):
    """
    XRF55 Dataset (Subset for 8 actions and all users)

    Format requirements:
    - Files are in .npy format.
    - Filenames follow the pattern a_b_c.npy
      a = User ID (2 digits)
      b = Action ID (2 digits) -> will be zero-indexed for PyTorch
      c = Repetition number

    Split rule is determined by a maximum repetition number for training.
    """
    def __init__(
        self,
        root_dir,
        split="train",
        case_split=0,
        split_mode=DEFAULT_SPLIT_MODE,
        shape_option="2d",
        num_sub=30,  # XRF55 typically uses Intel 5300 with 30 subcarriers per antenna
        normalize=True,
        norm_type="minmax",
        mean=None,
        std=None,
        target_time=None,
        allowed_labels=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.case_split = case_split
        self.split_mode = _check_split_mode(split_mode)
        self.shape_option = shape_option
        self.num_sub = num_sub
        self.normalize = normalize
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.target_time = target_time
        self.allowed_labels = tuple(DEFAULT_SELECTED_LABELS) if allowed_labels is None else tuple(allowed_labels)
        self.label_map = {label: idx for idx, label in enumerate(self.allowed_labels)}

        self.samples = []
        self.build_index()

    def build_index(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        for file_path in find_xrf55_wifi_files(self.root_dir):
            name_parts = file_path.stem.split('_')
            
            # Ensure it matches a_b_c format
            if len(name_parts) == 3:
                user_id, action_id, rep_id = name_parts
                rep_num = int(rep_id)
                
                action_id_int = int(action_id)
                if action_id_int not in self.label_map:
                    continue

                if self.split_mode == "position":
                    raise ValueError("split_mode='position' is only available for SSHAR")
                if self.split_mode == "repeat":
                    # Chia ngau nhien 7:1:1 theo lop (xem _assign_splits)
                    held_out = False
                else:
                    # Chia theo nguoi
                    user_num = int(user_id)
                    held_out = user_num not in USER_CASES["xrf55"][self.case_split]

                # Remap selected labels to contiguous indices 0..5.
                label = self.label_map[action_id_int]

                self.samples.append({
                    "path": str(file_path),
                    "label": label,
                    "rep": rep_num,
                    "held_out": held_out,
                })

        splits = _assign_splits(
            [s["label"] for s in self.samples],
            [s["held_out"] for s in self.samples],
            carve_test=self.split_mode == "repeat",
        )
        self.samples = [s for s, sp in zip(self.samples, splits) if sp == self.split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = np.load(sample["path"]).astype(np.float32)
        y = sample["label"]
        
        # Reshape logically based on shape_option
        if self.shape_option == "3d":
            # Assuming raw input is 2D: (C, Time) where C = dev_anten * num_sub
            if len(x.shape) == 2:
                c, t = x.shape
                assert c % self.num_sub == 0
                dev_anten = c // self.num_sub
                x = x.reshape(dev_anten, self.num_sub, t)
        elif self.shape_option == "2d":
            # If input is already 3D: (dev_anten, sub, Time), flatten it back to 2D
            if len(x.shape) == 3:
                d, s, t = x.shape
                x = x.reshape(d * s, t)

        if self.normalize:
            if self.norm_type == "minmax":
                xmin = x.min()
                xmax = x.max()
                x = (x - xmin) / (xmax - xmin + 1e-8)
            elif self.norm_type == "zscore":
                if self.mean is None or self.std is None:
                    raise ValueError("mean/std required for zscore")
                x = (x - self.mean) / (self.std + 1e-8)
            else:
                raise ValueError(f"Unknown norm_type {self.norm_type}")

        x = torch.from_numpy(x).float()

        if self.target_time is not None and x.ndim in (2, 3) and x.shape[-1] != self.target_time:
            x = F.adaptive_max_pool1d(x, output_size=self.target_time)

        return x, y


# ============================================================
# Compute XRF55 mean/std
# ============================================================
def compute_xrf55_mean_std(root_dir, case_split, shape_option="2d", num_sub=30, target_time=None,
                           split_mode=DEFAULT_SPLIT_MODE):
    """
    Compute global mean/std for XRF55 using TRAIN split only.
    """
    dataset = XRF55Dataset(
        root_dir=root_dir,
        split="train",
        case_split=case_split,
        split_mode=split_mode,
        shape_option=shape_option,
        num_sub=num_sub,
        normalize=False,
        target_time=target_time,
    )

    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    n = len(dataset)
    print(f"[XRF55] computing mean/std over {n} train samples ...")
    for i, (x, _) in enumerate(dataset):
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"[XRF55]   mean/std {i + 1}/{n}")
        x = x.numpy().astype(np.float64)
        total_sum += x.sum()
        total_sq += np.square(x).sum()
        total_count += x.size

    if total_count == 0:
        return 0.0, 1.0

    mean = total_sum / total_count
    std = np.sqrt(total_sq / total_count - mean ** 2)

    print(f"XRF55 mean = {mean:.6f}")
    print(f"XRF55 std  = {std:.6f}")

    return mean, std


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=DEFAULT_PIN_MEMORY,
):

    def make(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return make(train_dataset, True), make(val_dataset, False), make(test_dataset, False)


# ============================================================
# Unified Loader
# ============================================================
def load_dataset(
    name,
    root_dir,
    batch_size=DEFAULT_BATCH_SIZE,
    normalize=True,
    norm_type=None,
    **kwargs
):
    """
    Build train / val / test loaders.

    Split:
      SSHAR repeat    -> protocol 3B, 7:1:2 by repetition (4480 / 640 / 1280)
      SSHAR user      -> protocol 3A, 7 / 1 / 2 subjects (case 0..4)
      SSHAR position  -> test = held-out positions, rest train:val = 7:1
      XRF55 repeat    -> 7:1:1 stratified by class (fixed seed)
      XRF55 user      -> test = held-out users, rest train:val = 7:1
      ut_har          -> the dataset's own X_train / X_val / X_test files

    The best epoch is chosen on val; test is only used for the final report.

    kwargs: case_split, split_mode, shape_option, num_sub (XRF55),
            target_time, intel_shape, allowed_labels.
    """

    name = name.lower()
    shape_option = kwargs.get("shape_option", "2d")
    num_sub = kwargs.get("num_sub", 30) # Used for XRF55
    target_time = kwargs.get("target_time", None)
    intel_shape = kwargs.get("intel_shape", False)
    allowed_labels = kwargs.get("allowed_labels", DEFAULT_SELECTED_LABELS)
    case_split = kwargs.get("case_split", 0)
    split_mode = _check_split_mode(kwargs.get("split_mode", DEFAULT_SPLIT_MODE))
    ratio = ":".join(map(str, SPLIT_RATIO))
    sshar = name.lower().startswith("sshar")
    if split_mode == "user" and sshar:
        users = USER_CASES["sshar"][case_split]
        val_u = SSHAR_VAL_SUBJECTS[case_split]
        test_u = sorted(set(SSHAR_ALL_SUBJECTS) - set(users))
        print(f"Split mode = user (protocol 3A, 7:1:2), case = {case_split} | "
              f"train: {sorted(set(users) - {val_u})} | val: [{val_u}] | test: {test_u}")
    elif split_mode == "repeat" and sshar:
        print("Split mode = repeat (protocol 3B, 7:1:2: dir00 rep1-7/8/9-10; "
              "dir01/02 rep1-3 + rep4 of one dir / rep4 of the other dir / rep5)")
    elif split_mode == "user":
        print(f"Split mode = user, case = {case_split} (val = 1/8 of train users' samples)")
    elif split_mode == "position":
        if not 0 <= case_split < N_POSITION_CASES:
            raise ValueError(f"split_mode=position needs case_split in 0..{N_POSITION_CASES - 1}")
        print(f"Split mode = position, case = {case_split} "
              f"(test positions: pos % {N_POSITION_CASES} == {case_split}, "
              f"val = 1/8 of the other positions' samples)")
    else:
        print(f"Split mode = repeat (random {ratio} train:val:test, stratified by class, "
              f"seed={SPLIT_SEED})")

    # =======================================================
    # UT_HAR
    # =======================================================
    if name == "ut_har":
        if norm_type is None:
            norm_type = "minmax"

        mean = std = None
        if normalize and norm_type == "zscore":
            mean, std = compute_ut_har_mean_std(
                root_dir
            )

        def make(split):
            return UTHARDataset(
                root_dir=root_dir,
                split=split,
                normalize=normalize,
                norm_type=norm_type,
                mean=mean,
                std=std,
            )

    # =======================================================
    # SSHAR ESP / ASUS
    # =======================================================
    elif name in ("sshar_esp", "sshar_asus"):
        cls = SSHAR_ESP_Dataset if name == "sshar_esp" else SSHAR_ASUS_Dataset
        if norm_type is None:
            norm_type = "zscore"

        common = dict(
            root_dir=root_dir,
            case_split=case_split,
            split_mode=split_mode,
            shape_option=shape_option,
            target_time=target_time,
            intel_shape=intel_shape,
            allowed_labels=allowed_labels,
        )

        mean = std = None
        if normalize and norm_type == "zscore":
            # mean/std from the TRAIN split only
            mean, std = compute_sshar_mean_std(cls(split="train", normalize=False, **common))

        def make(split):
            return cls(
                split=split,
                normalize=normalize,
                norm_type=norm_type,
                mean=mean,
                std=std,
                **common,
            )

    # =======================================================
    # XRF55
    # =======================================================
    elif name == "xrf55":
        if norm_type is None:
            norm_type = "zscore"

        mean = std = None
        if normalize and norm_type == "zscore":
            mean, std = compute_xrf55_mean_std(root_dir, case_split, shape_option, num_sub, target_time,
                                               split_mode=split_mode)

        def make(split):
            return XRF55Dataset(
                root_dir=root_dir,
                split=split,
                case_split=case_split,
                split_mode=split_mode,
                shape_option=shape_option,
                num_sub=num_sub,
                normalize=normalize,
                norm_type=norm_type,
                mean=mean,
                std=std,
                target_time=target_time,
                allowed_labels=allowed_labels,
            )

    else:
        raise ValueError(
            f"Unknown dataset: {name}"
        )

    train_dataset = make("train")
    val_dataset = make("val")
    test_dataset = make("test")

    # =======================================================
    for split_name, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        if len(ds) == 0:
            raise RuntimeError(
                f"[{name}] {split_name} split is empty (0 samples). "
                f"Check --data_root='{root_dir}' (files must exist under it, "
                f"named as expected), the allowed labels {tuple(allowed_labels)}, "
                f"and the split mode (split_mode={split_mode}, "
                f"case_split={case_split})."
            )

    train_loader, val_loader, test_loader = create_dataloader(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
    )

    # Handling info extraction
    x, _ = train_dataset[0]
    input_shape = tuple(x.shape)

    num_classes = len(set(train_dataset.y)) if isinstance(train_dataset, UTHARDataset) else len(set(s["label"] for s in train_dataset.samples))

    info = {
        "dataset": name,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }

    return (
        train_loader,
        val_loader,
        test_loader,
        info,
    )
