import h5py
import numpy as np

trimsize = 40

qsqlist = np.array([0.00001, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0])

with h5py.File("../res_Bs.h5", 'r') as f, h5py.File("../res_Bs_re.h5",'w') as f2:
    arr = np.array([x[:trimsize] for x in f['results']])
    f2.create_dataset('results', data=arr)
    f2.create_dataset('qsqlist', data=qsqlist)


