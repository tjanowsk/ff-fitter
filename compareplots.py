import json
import h5py
import dpath
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

indir = 'input/'
resdir = 'results/'

inputfiles = [x for x in glob(indir+'*') if 'star' not in x]

dinp = {}
dal = {}

for inp in inputfiles:
    with open(inp,'r') as f:
        dinp[inp] = json.load(f)     
    alp = inp.replace(indir,resdir).replace('.json','/alpha')
    with open(alp,'r') as f:
        f.readline()
        dal[inp] = list(map(float, f.readline()[:-2].split(',')))

#mBs = [ [x['name'] for x in FF] for (path,FF) in dpath.util.search(dinp,'*.json/*/FF',yielded=True)]

def getdata(FFname, inp):
    for infile in inp:
        for h5file in inp[infile]:
            FFlist = [x['name'] for x in inp[infile][h5file]['FF']]
            if FFname in FFlist:
                mpole = [x['m_pole'] for x in inp[infile][h5file]['FF'] if x['name'] == FFname][0]
                print(infile, h5file, FFlist, mpole)
                break
        else:
            continue
        break
    idx = FFlist.index(FFname)
    with h5py.File(h5file,'r') as f:
        qsqlist = f['qsqlist'][...]
        l = len(qsqlist)
        samples = [ sample[idx*l:(idx+1)*l] for sample in f['results'][...]]
    return(qsqlist,np.mean(samples,axis=0), np.std(samples,axis=0))

def getalphas(FFname, inp, als):
    dres = {}
    for infile in inp:
        alists = [inp[infile][h5file]['alphalist'] for h5file in inp[infile]]
        numalphas = [inp[infile][h5file]['num_pars'] for h5file in inp[infile]]
        FFlist = list(np.concatenate([[x['name'] for x in inp[infile][h5file]['FF']] for h5file in inp[infile] ],axis=0))
        idx = FFlist.index(FFname)
        FFlen = [len([x['name'] for x in inp[infile][h5file]['FF']]) for h5file in inp[infile] ]
        numalphas2 = np.concatenate([ [x]*y for x,y in zip(numalphas,FFlen) ], axis=0)
        #print(numalphas2)
        lb = sum(numalphas2[:idx])
        ub = lb + numalphas2[idx]
        #print(FFlen)
        alist = [np.array( eval(a.replace('None', 'list(range(fl*n))').replace('NA','n'))) for a,n,fl in zip(alists,numalphas,FFlen) ]
        for i in range(len(alist) - 1):
            alist[i+1] = alist[i+1] + max(alist[i]) + 1
        dres[infile] = [als[infile][i] for i in np.concatenate(alist)[lb:ub]]
       # print(alists,numalphas)
    return dres

mV = 0.77
mB = 5.28
mpole = 5.325
ffname = 'Vperp_d'

def z(qsq):
    tplus = (mB+mV)**2
    tminus =(mB-mV)**2
    t0 = tplus*(1-np.sqrt(1-tminus/tplus))
    return (np.sqrt(tplus-qsq)-np.sqrt(tplus-t0))/(np.sqrt(tplus-qsq) + np.sqrt(tplus-t0))

def poly(z, *alpha):
    ''' Returns a polynomial \alpha_i z^i'''
    res = 0
    for i, an in enumerate(alpha):
        print(an,i)
        res += an*z**i
    return res

def fun(qsq, alpha, pole):
    return poly(z(qsq)-z(0), *alpha)/(1-qsq/pole**2)

dalres = getalphas('Vperp_d', dinp, dal)

xv = np.arange(-5,20,0.1)
for file in dalres:
#    print(file)
    plt.plot(xv, fun(xv, dalres[file], mpole), label = file.replace('input/','').replace('.json',''))

plt.errorbar(*getdata('Vperp_d',dinp), fmt='.', label='data')
plt.legend()
plt.show()

##FFlist = [ [ x['FF'] for x in dinp[infile] ] for infile in dinp ]
#FFlist = [np.concatenate([ [FF['name'] for FF in dinp[infile][file]['FF'] ] for file in dinp[infile] ]) for infile in dinp ]
#print(FFlist[0])
