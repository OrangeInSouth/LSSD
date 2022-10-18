import json
import numpy as np

data_path = "../experimental_results/diverse_ted8_m2o/soft_selective25/soft_sentence_od_output.json"


f = open(data_path)
data = json.load(f)
f.close()

variances = []
for lang, v in data.items():
    var_for_lang = []
    for e in v:
        if len(e) > 0:
            var_for_lang.append(np.array(e).var())
    variances.append(np.array(var_for_lang).mean())

print("M2O:")
print(sum(variances) / len(variances))
print()

data_path = "../experimental_results/diverse_ted8_o2m/soft_LA_od05/soft_sentence_od_output.json"
f = open(data_path)
data = json.load(f)
f.close()

variances = []
for lang, v in data.items():
    var_for_lang = []
    for e in v:
        if len(e) > 0:
            var_for_lang.append(np.array(e).var())
    variances.append(np.array(var_for_lang).mean())

print("O2M:")
print(sum(variances) / len(variances))
print()