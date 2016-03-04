from numpy import *
from scipy.sparse.linalg import eigsh
import copy,time,pdb,warnings

from blockmatrix.blocklib import eigbsh,eigbh,get_bmgen,tobdmatrix
from rglib.hexpand import NullEvolutor

__all__=['get_H','get_H_bm']

def get_H(H,hgen):
    '''
    Directly solve the ground state energy through lanczos.

    Parameters:
        :H: <OpCollection>, the hailtonian operator.
        :hgen: <RGHGen>, the hamiltonian generator.

    Return:
        matrix, the hamiltonian matrix.
    '''
    nsite=H.nsite
    hgen=copy.deepcopy(hgen)
    if not isinstance(hgen.evolutor,NullEvolutor):
        raise ValueError('The evolutor must be null!')

    for i in xrange(nsite):
        ops=H.query(i)
        intraop,interop=[],[]
        for op in ops:
            siteindices=array(op.siteindex)
            if any(siteindices>i):
                interop.append(op)
            else:
                intraop.append(op)
        hgen.expand(intraop)
        hgen.trunc()
    return hgen.H

def get_H_bm(H,hgen,bstr):
    '''
    Get the hamiltonian with block marker.

    Parameters:
        :H: <OpCollection>, the hailtonian operator.
        :hgen: <RGHGen>, the hamiltonian generator.

    Return:
        matrix, the hamiltonian matrix.
    '''
    bmgen=get_bmgen(hgen.spaceconfig,bstr)
    nsite=H.nsite
    for i in xrange(nsite):
        ops=H.filter(lambda sites:all(sites<=i) and (i in sites))
        hgen.expand(ops)
        bm=bmgen.update_blockmarker(hgen.block_marker,nsite=nsite)
        hgen.trunc(U=None,block_marker=bm)
    return hgen.H,bm

