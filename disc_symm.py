'''
Discrete symmetries.
'''

from numpy import *
from scipy import sparse as sps
import pdb

from rglib.mps import OpUnit

__all__=['DiscSymm','PHSymm','FlipSymm','C2Symm']

class DiscSymm(object):
    '''
    Discrete symmetrix class.

    Attributes:
        :name: str, the name of this symmetrc.
        :proj: number, the projection matrices.
    '''
    def __init__(self,name,proj=None):
        self.name=name
        self._proj=proj

    def get_projector(self,parity=None):
        '''
        get the specific projector.
        '''
        assert(parity in [1,-1,None])
        if parity is None:
            return self._proj
        else:
            return 0.5*(sps.identity(self._proj.shape[0])+parity*self._proj)

    def act_on_state(self,phi):
        '''
        Act on the specific state.

        Parameters:
            :phi: 1D array, the state vector.

        Return:
            1D array, the state after parity action.
        '''
        P=self.get_projector()
        return P.dot(phi)

    def act_on_op(self,op):
        '''
        Act on the specific operator.

        Parameters:
            :op: matrix, the operator.

        Return:
            matrix, the operator after parity action.
        '''
        P=self.get_projector().tocsr()
        return P.dot(op).dot(P.T.conj())

    def project_state(self,phi,parity,**kwargs):
        '''
        project out the specific parity from a state.

        Parameters:
            :phi: 1D array, the state vector.
            :parity: 1/-1, the specific parity.

        Return:
            1D array, the state after projection.
        '''
        assert(parity==1 or parity==-1)
        P=self.get_projector(parity).tocsr()
        return P.dot(phi)

    def project_op(self,op,parity,**kwargs):
        '''
        project out the specific parity from an operator.

        Parameters:
            :op: matrix, the operator.
            :parity: 1/-1, the specific parity.

        Return:
            matrix, the operator after project action.
        '''
        assert(parity==1 or parity==-1)
        P=self.get_projector(parity).tocsr()
        return P.dot(op.tocsc()).dot(P.T.conj())

    def check_op(self,op):
        '''
        Check a specific op if it do obey this symmetry.
        '''
        P=self.get_projector().tocsr()
        nop=P.dot(op.tocsc()).dot(P.T.conj())
        diff=(nop-op).data
        print diff
        return allclose(diff,0)

    def check_parity(self,phi,**kwargs):
        '''
        check the parity of a state.

        Parameters:
            :phi: 1D array, the state vector.

        Return:
            len-2 tuple, the even-odd component.
        '''
        comps=[]
        for parity in [-1,1]:
            comp=self.project_state(phi,parity,**kwargs)
            comps.append(comp.dot(phi.conj()))
        return comps

    def update(self,**kwargs):
        '''
        Update this symmetry projection operator.
        '''
        raise NotImplementedError()

class PHSymm(DiscSymm):
    '''
    Particle hole symmetry.
    '''
    def __init__(self,proj=None):
        super(PHSymm,self).__init__('ph',proj)

    def J(self,i):
        '''
        Get the transformation opunit at specific site.

        Parameters:
            :i: int, the site index.
        '''
        data=zeros([4,4],dtype='int32')
        data[0,3]=-1
        data[3,0]=1
        data[1,1]=(-1)**i
        data[2,2]=(-1)**i
        ou=OpUnit(label='J',siteindex=i)
        return ou

class FlipSymm(DiscSymm):
    '''
    Spin flip symmetry.
    '''
    def __init__(self,proj=None):
        super(FlipSymm,self).__init__('sf',proj)

    def P(self,i):
        '''
        Get the transformation opunit at specific site.

        Parameters:
            :i: int, the site index.
        '''
        data=zeros([4,4],dtype='int32')
        data[0,0]=1
        data[3,3]=-1
        data[1,2]=1
        data[2,1]=1
        ou=OpUnit(label='P',siteindex=i)
        return ou

class C2Symm(DiscSymm):
    '''
    space left-right reflection symmetry.
    '''
    def __init__(self,proj=None):
        super(C2Symm,self).__init__('c2',proj)

    def update(self,nl,nr,**kwargs):
        '''
        update the projection matrix.
        '''
        NL=len(nl)
        NR=len(nr)
        assert(NL==NR)
        yindices=arange(NL*NR)
        xindices=NL*(yindices%NR)+yindices/NR
        factor=(-1)**((nl[:,newaxis]*nr)%2).ravel()
        self._proj=sps.coo_matrix((factor,(xindices,yindices)),shape=(NL*NR,NL*NR),dtype='int32')

