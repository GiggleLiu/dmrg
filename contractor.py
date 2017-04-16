'''
Constractor for vmps.
'''

from pymps import Tensor
from numpy import *

__all__=['Contractor']

class Contractor(object):
    '''
    Contraction handler.

    Attributes:
        :mpo/ket: <MPO>/<MPS>,the operator and ket.
        :bra_labels: list, the labels for bra, 
        :LPART/RPART: list of <Tensor>, the result of contraction from left/right.

    Readonly Attributes:
        :bra: <MPS>, bra is the hermitian conjugate of ket.
    '''
    def __init__(self,mpo,ket,bra_bond_str='t'):
        #check validity
        if bra_bond_str==ket.labels[1] or bra_bond_str==mpo.labels[2] or mpo.labels[2]==ket.labels[1]: raise ValueError('Same bond string for bra, mpo and ket.')
        if mpo.labels[1]!=ket.labels[0]: raise ValueError('Site string for mpo and ket do not match!')

        self.bra_labels=[mpo.labels[0],bra_bond_str]
        self.ket=ket
        self.mpo=mpo

        #initialize datas in LPART and RPART
        nsite=self.ket.nsite
        initial_data=Tensor(ones([1,1,1]),labels=[self.bra_labels[1]+'_0',self.mpo.get(0).labels[0],self.ket.get(0).labels[0]])
        self.LPART=[initial_data]
        initial_data=Tensor(ones([1,1,1]),labels=['%s_%s'%(self.bra_labels[1],nsite),self.mpo.get(nsite-1).labels[-1],self.ket.get(nsite-1).labels[-1]])
        self.RPART=[initial_data]

    @property
    def bra(self):
        return self.ket.tobra(labels=self.bra_labels)

    def lupdate(self,i):
        '''
        Update LPARTs.

        Parameters:
            :i: int, the target length to update.
        '''
        if i>len(self.LPART) or i>self.ket.l:  #illgal move
            raise ValueError('Illegal move!')
        FL=self.LPART[i-1]
        cket=self.ket.get(i-1)
        bs,bb=self.bra_labels
        cbra=cket.conj().make_copy(['%s_%s'%(bb,i-1),'%s_%s'%(bs,i-1),'%s_%s'%(bb,i)],copydata=False)
        cmpo=self.mpo.get(i-1)
        FL=cbra*FL*cmpo*cket
        if i==len(self.LPART): #gen new terms.
            self.LPART.append(FL)
        else:
            self.LPART[i]=FL
            self.LPART=self.LPART[:i+1]

    def rupdate(self,i):
        '''
        Update RPARTs.

        Parameters:
            :i: int, the target length to update.
        '''
        nsite=self.ket.nsite
        l=nsite-i
        if i>len(self.RPART) or l<self.ket.l:  #illgal move
            raise ValueError('Illegal move!')
        FR=self.RPART[i-1]
        cket=self.ket.get(l)
        bs,bb=self.bra_labels
        cbra=cket.conj().make_copy(labels=['%s_%s'%(bb,l),'%s_%s'%(bs,l),'%s_%s'%(bb,l+1)],copydata=False)
        cmpo=self.mpo.get(l)
        FR=cbra*FR*cmpo*cket
        if i==len(self.RPART): #gen new terms.
            self.RPART.append(FR)
        else:
            self.RPART[i]=FR
            self.RPART=self.RPART[:i+1]

    def evaluate(self):
        '''
        Get the expectation value of mpo.
        '''
        nsite=self.ket.nsite
        bra=self.bra
        attach_S='A' if self.ket.l==nsite else 'B'
        FL=Tensor(ones([1,1,1]),labels=[bra.get(0,attach_S=attach_S).labels[0],self.mpo.get(0).labels[0],self.ket.get(0,attach_S=attach_S).labels[0]])
        for i in xrange(nsite):
            cbra=bra.get(i,attach_S=attach_S)
            cket=self.ket.get(i,attach_S=attach_S)
            ch=self.mpo.get(i)
            FL=cbra*FL*ch*cket
        return FL.item()

    def show(self,space=2):
        '''Show the contraction Graphically.'''
        nitem=self.noperand
        for i,item in enumerate(self.operands):
            item.show(offset=(0,(nitem-i)*space))

    def contract2l(self):
        '''Contract all available LPART and RPART.'''
        l=self.ket.l
        for i in xrange(1,l):
            self.lupdate(i)
        for i in xrange(1,self.ket.nsite-l+1):
            self.rupdate(i)


