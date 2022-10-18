
import json
from collections import Counter
uniform_path = "../experimental_results/sub_wmt4_m2o/uniform_128K/LS_valid_loss_history.json"
LSSD_path = "../experimental_results/sub_wmt4_m2o/od10_128K/LS_valid_loss_history.json"

f = open(uniform_path)
base_data = json.load(f)
f.close()

f = open(LSSD_path)
LSSD_data = json.load(f)
f.close()

print("baseline:")
for lang, loss in base_data.items():
    count = 0
    counter = Counter()
    for i in range(1, len(loss)):
        if loss[i] < loss[i-1] or i == len(loss) - 1:
            counter.update([count])
            count = 0
        else:
            count += 1
    res = sorted(list(counter.items()), key=lambda x: x[0])
    print(lang, res)

print("LSSD:")
for lang, loss in LSSD_data.items():
    count = 0
    counter = Counter()
    for i in range(1, len(loss)):
        if loss[i] < loss[i-1] or i == len(loss) - 1:
            counter.update([count])
            count = 0
        else:
            count += 1
    res = sorted(list(counter.items()), key=lambda x: x[0])
    print(lang, res)