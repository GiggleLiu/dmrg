from numpy import *
from scipy.sparse.linalg import eigsh
import copy,time,pdb,warnings

from blockmatrix.blocklib import eigbsh,eigbh,get_bmgen,tobdmatrix
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
    bmgen=get_bmgen(hgen.spaceconfig,bstr)
    nsite=hgen.nsite
    for i in xrange(nsite):
        hgen.expand1()
        bm=bmgen.update_blockmarker(hgen.block_marker,nsite=nsite)
        hgen.trunc(U=None,block_marker=bm)
    return hgen.H,bm

