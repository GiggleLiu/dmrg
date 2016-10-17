from numpy import *
from scipy.sparse.linalg import eigsh
import scipy.sparse as sps
import copy,time,pdb,warnings

from blockmatrix.blocklib import eigbsh,eigbh,SimpleBMG,tobdmatrix
from rglib.hexpand import NullEvolutor

__all__=['get_H','get_H_bm']

def get_H(hgen):
    '''
    Directly solve the ground state energy through lanczos.

    Parameters:
        :hgen: <RGHGen>, the hamiltonian generator.

    Return:
        matrix, the hamiltonian matrix.
    '''
    nsite=hgen.nsite
    hgen=copy.deepcopy(hgen)
    if not isinstance(hgen.evolutor,NullEvolutor):
        raise ValueError('The evolutor must be null!')

    for i in xrange(nsite):
        hgen.expand1()
        hgen.trunc(U=None)
    return hgen.H

def get_H_bm(hgen,bstr):
    '''
    Get the hamiltonian with block marker.

    Parameters:
        :hgen: <RGHGen>, the hamiltonian generator.

    Return:
        matrix, the hamiltonian matrix.
    '''
    bmgen=SimpleBMG(spaceconfig=hgen.spaceconfig,qstring=bstr)
    nsite=hgen.nsite
    for i in xrange(nsite):
        hgen.expand1()
        bm,pm=bmgen.update1(hgen.block_marker)
        hgen.trunc(U=sps.coo_matrix((ones(bm.N),(pm,arange(bm.N))),dtype='int32'),block_marker=bm)
    return hgen.H,bm

