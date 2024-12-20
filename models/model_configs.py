import os
import json

model_configs = {
    "vit_custom": {
        "patch_size": 8,
        "dim": 128,
        "depth": 8,
        "heads": 4,
        "mlp_dim": 128,
    },
    "vit_tiny": {
        "patch_size": 4,
        "dim": 192,
        "depth": 12,
        "heads": 3,
        "mlp_dim": 768,
    },
    "vit_small": {
        "patch_size": 4,
        "dim": 384,
        "depth": 12,
        "heads": 6,
        "mlp_dim": 1536,
    },
    "vit_base": {
        "patch_size": 8,
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "mlp_dim": 3072,
    },
}

file = "model_configs.json"
with open(file, "w") as f:
    json.dump(model_configs, f, indent=4)
