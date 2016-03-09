'''
Utility for Projecting a specific MPS to the current iteration.
'''

from numpy import *

from rglib.mps import tensor

class Projector(object):
    '''
    Project handler.

    Attibutes:
        :mps: <MPS>, the matrix product state to evolve with hamiltonian.
        :L/R: <Tensor>, the left tensor and right tensor for contraction.
    '''
    def __init__(self,mps):
        self.mps=mps
        if self.mps.labels!=['s','a']:
            self.mps=self.mps.toket(labels=['s','a'])
        self.L=None
        self.R=None

    @property
    def nsite(self):
        '''The number fo sites'''
        return self.mps.nsite

    def initialize(self,hgen_l,hgen_r):
        '''
        Initialize the projector, calculation the first state projection.

        Parameters:
            :hgen_l/hgen_r: <RGHGen>, starting hamiltonian generator for left and right block with `NL+NR = nsite-2`.

        Return:
            ndarray, the representation of mps <al,sl,sl+1,al+2|self.mps>.
        '''
        LL=hgen_l.evolutor.get_AL(dense=True)
        RL=hgen_r.evolutor.get_AL(dense=True)
        nsite=self.nsite
        assert(len(LL)+len(RL)==nsite-2)
        L=tensor.Tensor(ones([1,1]),labels=['b_0,a_0'])
        R=tensor.Tensor(ones([1,1]),labels=['b_%s,a_%s'%(nsite,nsite)])
        for i,bi in enumerate(LL):
            bi=tensor.Tensor(bi,labels=['s_%s'%i,'b_%s'%i,'b_%s'%(i+1)])
            ai=self.mps.get(i,attach_S='B')
            L=tensor.contract([L,bi,ai])
        for i,bi in enumerate(RL):
            li=nsite-i-1
            bi=tensor.Tensor(bi.conj(),labels=['s_%s'%li,'b_%s'%(li+1),'b_%s'%li])
            ai=self.mps.get(li,attach_S='B')
            R=tensor.contract([R,bi,ai])
        res=tensor.contract([L,R])

    def update(self.hgen_l,hgen_r):
        '''
        Update the projector,
        '''
        pass
