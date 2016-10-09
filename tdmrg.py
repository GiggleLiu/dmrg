'''
Time dependant DMRG, including TEBD, tDMRG.
'''

from numpy import *
from scipy.linalg import expm

from rglib import RGHGen
from lanczos import get_H

class EvolveOp(object):
    '''
    Evolusion Operator.

    Attributes:
        :opc: <OpCollection>, the hamiltonian kernel for time evolution.
    '''
    def __init__(self,opc):
        self.opc=opc

    def get_even_eop(self,spaceconfig):
        '''Get the evolution operator collection for even bonds.'''
        nsite=self.opc.nsite
        #query even bonds, like 1-2 or 1.
        res=None
        for i in xrange(1,nsite,2):
            bop=self.opc.query(i,i+1)
            sop=self.opc.query(i)
            op=bop+sop
            #get the hamiltonian
            hgen=RGHGen(spaceconfig=spaceconfig,H=op,evolutor_type='null')
            H=get_H(hgen)
            #get the evolution matrix, exponential of H
            expH=expm(H)
            #SVD decompose to make it into .
        return res

    def get_odd_eop(self):
        '''Get the evolution operator collection for odd bonds.'''
        pass
