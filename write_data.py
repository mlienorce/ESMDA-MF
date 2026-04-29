import csv
import pathlib
import numpy as np
import pickle
import pandas as pd
import h5py
from ThreeDGiGEarth.common import h5_to_dict
from pathlib import Path
from collections import defaultdict
#PROJECT_ROOT =  pathlib.Path(__file__).resolve().parents[1]

data_index = [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]

datatyp = 'Bfield'

BENCHMARK_ROOT = '/home/AD.NORCERESEARCH.NO/mlie/3DGiG/Jacobian/inversion/data/Benchmark-3/'
REFERENCE_MODEL = '/home/AD.NORCERESEARCH.NO/mlie/3DGiG/Jacobian/inversion/data/Benchmark-3/globalmodel.h5'

#REFERENCE_MODEL = PROJECT_ROOT / 'inversion' / 'data' / 'Benchmark-3' / 'globalmodel.h5'
#BENCHMARK_ROOT = PROJECT_ROOT / 'inversion-NN' / 'data' / 'Benchmark-3'

with h5py.File(REFERENCE_MODEL, 'r') as f:
    ref_model = h5_to_dict(f)

tvd = ref_model['wellpath']['Z'][:, 0] * 3.28084  # feet

#ED = T[:,5]

#with open('../inputs/trajectory.DAT','r') as f:
#    lines = f.readlines()
#    tvd = [el.strip().split()[1] for el in lines][1:]

k = open('assim_index.csv','w',newline='')
#writer4 = csv.writer(k)
l = open('datatyp.csv','w',newline='')
writer5 = csv.writer(l)

data = {}
var={}

# build a pandas dataframe with the data.
# The tvd is the index and the tuple (freq,dist) is the columns
abs_frac = 0.025
min_floor = 1e-12
ampl_fac = 1.0  # if you still need abs floor logic; otherwise ignore

# First pass: accumulate sums and counts per step index
sums_by_step = defaultdict(lambda: None)   # maps step_idx -> sum array
counts_by_step = defaultdict(int)

for di in data_index:
    freq, dist = di
    p = Path(f'{BENCHMARK_ROOT}/logdata/{datatyp}_{dist}_{freq}.las')
    if not p.exists():
        continue
    with p.open('r') as f:
        lines = f.readlines()[1:]
        for step_idx, el in enumerate(lines):
            vals = np.array(el.strip().split()[1:], dtype=np.float64)
            if sums_by_step[step_idx] is None:
                sums_by_step[step_idx] = np.zeros_like(vals)
            sums_by_step[step_idx] += np.abs(vals)   # absolute values
            counts_by_step[step_idx] += 1

if not counts_by_step:
    raise RuntimeError("No data found to compute per-step means")

# Compute mean across all tool settings & data types for each step and measurement
max_step = max(sums_by_step.keys())
means_by_step = {}
for step in range(max_step + 1):
    if counts_by_step.get(step, 0) == 0:
        raise RuntimeError(f"No data for step {step}")
    means = sums_by_step[step] / counts_by_step[step]
    means = np.maximum(means, min_floor)   # avoid zero
    means_by_step[step] = means            # shape (n_meas,)

# Optional: small absolute floor based on global minima if you still want an abs floor
# (compute across all steps & channels if desired)
# global_min = min(np.min(sums_by_step[s] / counts_by_step[s]) for s in means_by_step)
# abs_floor = global_min * ampl_fac

# Second pass: populate data and set relative-variance per row using the step mean
for di in data_index:
    freq, dist = di
    try:
        p = Path(f'{BENCHMARK_ROOT}/logdata/{datatyp}_{dist}_{freq}.las')
        with p.open('r') as f:
            lines = f.readlines()
            values = [np.array(el.strip().split()[1:], dtype=np.float32) for el in lines[1:]]
            data[(freq, dist)] = values

            # Build var entries: for each row (step_idx) use the step mean to compute variance
            per_row_vars = []
            for step_idx, val in enumerate(values):
                mean_step = means_by_step[step_idx]               # shape (n_meas,)
                abs_err = abs_frac * mean_step                    # absolute error per channel for that step
                abs_err = np.maximum(abs_err, min_floor)         # avoid zero
                var_row = (abs_err ** 2).tolist()                # variance per channel for that row
                per_row_vars.append(var_row)

            # Choose structure expected downstream:
            # Option A: single header then per-row lists: [['ABS'] + [var_row1, var_row2, ...]]
            var[(freq, dist)] = [['ABS'] + per_row_vars]

            # Option B (if downstream expects ['ABS', [v0,v1,...]] same for all rows):
            # var[(freq, dist)] = ['ABS', list((abs_frac * np.mean(list(means_by_step.values()), axis=0))**2)]

    except Exception:
        data[(freq, dist)] = None
        var[(freq, dist)] = None
        continue




df = pd.DataFrame(data,columns=data_index,index=tvd)
df.index.name = 'tvd'
#df.to_csv('data.csv',index=True)
df.to_pickle('data.pkl')

df = pd.DataFrame(var,columns=data_index,index=tvd)
df.index.name = 'tvd'
with open('var.pkl','wb') as f:
    pickle.dump(df,f)

#filt = [i*10 for i in range(50)]
for c,_ in enumerate(tvd):
    #if c in filt:
    k.writelines(str(c) + '\n')
k.close()

writer5.writerow([str(el) for el in data_index])
l.close()
