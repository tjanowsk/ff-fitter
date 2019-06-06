import h5py
import numpy as np

trimsize = 40

with h5py.File("../res_Bs.h5", 'r') as f, h5py.File("../res_Bs_re.h5",'w') as f2:
    arr = np.array([x[:trimsize] for x in f['results']])
    f2.create_dataset('results',data=arr)

