"""CLINC150: 150 customer-assistant intents plus out-of-scope messages (CC BY 3.0)."""
from __future__ import annotations

import numpy as np
from datasets import load_dataset


def load():
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    oos = names.index("oos")
    split = lambda s: (list(ds[s]["text"]), np.array(ds[s]["intent"]))
    return split("train"), split("validation"), split("test"), names, oos
