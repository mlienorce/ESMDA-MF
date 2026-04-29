import csv
import pathlib
import numpy as np
import pickle
import pandas as pd
import h5py
from ThreeDGiGEarth.common import h5_to_dict

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

# build a pandas dataframe with the data.
# The tvd is the index and the tuple (freq,dist) is the columns


data = {}
var = {}
for di in data_index:
    freq, dist = di
    try:
        with open(f'{BENCHMARK_ROOT}/logdata/{datatyp}_{dist}_{freq}.las', 'r') as f:
            lines = f.readlines()
            values = [np.array(el.strip().split()[1:],dtype=np.float32) for el in lines[1:]]
            data[(freq, dist)] = values
            # Calculate mean for each measurement type (corresponding to data_order)
            # Convert to array for proper mean calculation across all lines
            values_array = np.array(values)  # shape (n_lines, 8)
            measurement_means = np.mean(np.abs(values_array), axis=0)  # shape (8,) - mean over all lines for each data_order element
            # Variance as (2.5% of mean)^2 for each measurement type
            var[(freq, dist)] = [['ABS'] + [[(0.025*measurement_means[idx])**2 for idx, el in enumerate(val)]] for val in values]
            
    except:
        data[(freq, dist)] = None
        var[(freq,dist)] = None
        pass

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
