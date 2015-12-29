'''
Utilities for Heisenberg model.
'''
from models import *
from matplotlib.pyplot import *

from mps.dmrg import DMRGEngine
from mps.hexpand.spinhexpand import SpinHGen
from tba.hgen import SpinSpaceConfig
from mps.vmps import VMPSEngine
from mps.core.mps import MPS

def heisenberg_vmps(J,Jz,h,nsite,append=False):
    '''
    Run vMPS for Heisenberg model.
    '''
    filename='mps_heisenberg_%s.dat'%(nsite)
    model=HeisenbergModel(J=J,Jz=Jz,h=h,nsite=nsite)
    if append:
        mps=MPS.load(filename)
    else:
        #run dmrg to get the initial guess.
        hgen=SpinHGen(spaceconfig=SpinSpaceConfig([2,1]))
        dmrgegn=DMRGEngine(hchain=model.H_serial,hgen=hgen,tol=0)
        dmrgegn.run_finite(endpoint=(1,'<-',0),maxN=40,tol=1e-12)
        #hgen=dmrgegn.query('r',nsite-1)
        mps=dmrgegn.get_mps(direction='<-')  #right normalized initial state
        mps.save(filename)

    #run vmps
    vegn=VMPSEngine(H=model.H,k0=mps)
    vegn.run()

def heisenberg_dmrg(J,Jz,h,nsite=10,which='finite',J2=0,J2z=None):
    '''
    Run iDMFT for heisenberg model.
    '''
    if J2==0:
        model=HeisenbergModel(J=J,Jz=Jz,h=h,nsite=nsite)
    else:
        model=HeisenbergModel2(J=J,Jz=Jz,J2=J2,J2z=J2z,h=h,nsite=nsite)
    hgen=SpinHGen(spaceconfig=SpinSpaceConfig([2,1]))
    dmrgegn=DMRGEngine(hchain=model.H_serial,hgen=hgen,tol=0)
    if which=='infinite':
        dmrgegn.run_infinite(maxiter=50,maxN=20,tol=0)
    elif which=='lanczos':
        dmrgegn.direct_solve()
    else:
        dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=[10,20,30,40,40],tol=0)
    pdb.set_trace()
