# %%
import os
import numpy as np

PATH_TO_RESULTS = "/home/huawei123/kwx1991442/code-classification/saved_data/metrics/baseline"

FILES = [file for file in os.listdir(PATH_TO_RESULTS) if "TEST_baseline" in file]
output = []
for file in FILES:
    with open(os.path.join(PATH_TO_RESULTS, file), "r") as f:
        output.append({m.split('=')[0] : float(m.split('=')[1]) for m in f.readline().split()})

output = {key : np.array([item[key] for item in output]) for key, _ in output[0].items()}

# output.keys()  # ['accuracy', 'mse', 'sq_corr_coef']

for key in output.keys():
    print(f"{key} : mean={output[key].mean():.2f} std={output[key].std():.2f}")

# %%
