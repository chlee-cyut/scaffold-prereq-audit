#!/usr/bin/env python3
"""Train LightGCN (seed 2020) on MOOCCube with temporal leave-one-out; writes the checkpoint in recbole/saved_t/
used by run_g2g3_t.py, scaffold_rl_v2.py, tau_sensitivity.py, and wilcoxon_test.py."""
import os
BASE = os.environ.get("SCAFFOLD_ROOT", "/workspace")
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)
import numpy as np
for _a,_t in [("float",float),("int",int),("bool",bool),("object",object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import json
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

config_dict = {
    "data_path": BASE + "/mooccube/recbole/dataset/",
    "checkpoint_dir": BASE + "/mooccube/recbole/saved_t/",
    "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
    "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "TIME_FIELD": "timestamp",
    "user_inter_num_interval": "[5,inf)", "item_inter_num_interval": "[5,inf)",
    # Temporal leave-one-out (each user's last = test, second-to-last = validation), full ranking
    "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO",
                  "mode": "full", "group_by": "user"},
    "metrics": ["Recall", "NDCG", "Hit", "Precision"], "topk": [5, 10, 20],
    "valid_metric": "NDCG@10",
    "embedding_size": 64, "n_layers": 2, "reg_weight": 1e-05,
    "epochs": 50, "train_batch_size": 8192, "eval_batch_size": 20000000,
    "stopping_step": 10, "learning_rate": 0.001,
    "device": "cuda", "gpu_id": 0, "seed": 2020, "reproducibility": True,
    "show_progress": False,
}
config = Config(model="LightGCN", dataset="mooccube_csmath_t", config_dict=config_dict)
init_seed(config["seed"], config["reproducibility"])
init_logger(config)
dataset = create_dataset(config)
print(dataset)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = LightGCN(config, train_data.dataset).to(config["device"])
trainer = Trainer(config, model)
best_score, best_valid = trainer.fit(train_data, valid_data, saved=True, show_progress=False)
test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
print("\n=== Test set (temporal LOO, full ranking) ===")
for k, v in test_result.items(): print(f"  {k}: {v}")
print("CHECKPOINT:", trainer.saved_model_file)
json.dump({"checkpoint": trainer.saved_model_file,
           "test_result": {k: float(v) for k, v in test_result.items()}},
          open(BASE + "/mooccube/lightgcn_t_result.json", "w"), indent=2)
