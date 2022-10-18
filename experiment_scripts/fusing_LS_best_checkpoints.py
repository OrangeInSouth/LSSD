"""
This scripts aims to fuse language-specific best checkpoints
"""

import json
import sys

import torch
from fairseq.file_io import PathManager
from scripts.average_checkpoints import average_checkpoints
# 1. get epochs of language-specific best checkpoints
checkpoint_path = sys.argv[1]

print(f"checkpoint path: {checkpoint_path}")

f = open(checkpoint_path + '/LS_valid_loss_history.json')
LS_valid_loss_history = json.load(f)
f.close()

LS_best_epochs = []

print('-'*26)
print(f"{'language pair':^15}|{'epoch':^10}")
for lang_pair, loss_history in LS_valid_loss_history.items():
    if '-' in lang_pair:
        print(f"{lang_pair:^15}|{loss_history.index(min(loss_history)) + 1:^10}")
        LS_best_epochs.append(loss_history.index(min(loss_history)) + 1)

average_loss = LS_valid_loss_history['all']
print(f"{'Average loss':^15}|{loss_history.index(min(average_loss)) + 1:^10}")
print('-'*26)

# 2. fusing language-specific best checkpoints
print("Info | start to fusing models")
inputs = [checkpoint_path + '/checkpoint' + str(epoch) + '.pt' for epoch in LS_best_epochs]
fused_model = average_checkpoints(inputs)
print("Info | fusion succeed!")

output_path = checkpoint_path + '/fusion_of_LS_best_checkpoints.pt'
print(f"Info | start to write fused model into {output_path}.")
with PathManager.open(output_path, "wb") as f:
    torch.save(fused_model, f)
print("Info | Writing succeed!")