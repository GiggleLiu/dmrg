'''
Discrete symmetries.
'''

from numpy import *
from scipy import sparse as sps
from scipy.linalg import norm
import pdb,copy

from rglib.mps import OpUnit
from tba.lattice import ind2c,c2ind
from rglib.hexpand.utils import kron_csr as kron

__all__=['DiscSymm','PHSymm','FlipSymm','C2Symm','SymmetryHandler']

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
        diff=(P.dot(op.tocsc())-op.tocsr().dot(P.tocsc())).data
        res=allclose(diff,0)
        if not res:
            pdb.set_trace()
        return res

    def get_parity(self,phi,**kwargs):
        '''
        get the parity of a state.

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

    def update(self,data):
        '''
        Update this symmetry projection operator.

        Parameters:
            :data: matrix, new data
        '''
        self._proj=data

class PHSymm(DiscSymm):
    '''
    Particle hole symmetry.
    '''
    def __init__(self,proj=None):
        super(PHSymm,self).__init__('J',proj)

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
        ou=OpUnit(label='J',data=data,siteindex=i)
        return ou

class FlipSymm(DiscSymm):
    '''
    Spin flip symmetry.
    '''
    def __init__(self,proj=None):
        super(FlipSymm,self).__init__('P',proj)
        data=zeros([4,4],dtype='int32')
        data[0,0]=1
        data[3,3]=-1
        data[1,2]=1
        data[2,1]=1
        ou=OpUnit(label='P',data=data)
        self._P=ou

    def P(self,i):
        '''
        Get the transformation opunit at specific site.

        Parameters:
            :i: int, the site index.
        '''
        ou=copy.copy(self._P)
        ou.siteindex=i
        return ou

class C2Symm(DiscSymm):
    '''
    space left-right reflection symmetry.
    '''
    def __init__(self,proj=None):
        super(C2Symm,self).__init__('C',proj)

    def update(self,n):
        '''
        update the projection matrix.

        Parameters:
            :n: 1D array, the number of electrons in left-right blocks.
        '''
        N=len(n)**2
        base=array([len(n)]*2)
        yindices=arange(N)
        yconfig=ind2c(yindices,base)
        xconfig=yconfig[:,array([1,0])]
        signs=(-1)**((n[yconfig[:,0]]*n[yconfig[:,1]])%2)
        xindices=c2ind(xconfig,base)
        self._proj=sps.coo_matrix((signs,(xindices,yindices)),shape=(N,N),dtype='int32')


class SymmetryHandler(object):
    '''
    The symmetry handler for DMRG iteration.

    Attributes:
        :target_sector: dict, the target sector {parity label, sector}
        :handlers: dict, the handlers.
        :useC: bool, use good quantum number C2.
        :detect_scope: integer, the number of lowest levels to get \
                in order to search for the lowest state with specific C2 parity.

    Note:
        keys for target_sector and handlers,
    
        * 'C', C2 space end-to-end symmetry.
        * 'J', Particle hole symmetry.
        * 'P', Spin flip symmetry.
    '''
    def __init__(self,target_sector,detect_scope=2):
        self.target_sector=target_sector
        self.handlers={}
        self.useC=True
        self.detect_scope=detect_scope
        for symm in target_sector:
            if symm=='C':
                self.handlers['C']=C2Symm()
            elif symm=='P':
                self.handlers['P']=FlipSymm()
            elif symm=='J':
                self.handlers['J']=PHSymm()
            else:
                raise ValueError('Unknow symmetry %s'%symm)

    @property
    def symms(self):
        '''Active discrete symmetries.'''
        res=self.handlers.keys()
        if not self.useC and 'C' in res:
            res.remove('C')
        return res

    def __eq__(self,target):
        if target==None:
            return len(self.target_sector)==0
        elif isinstance(target,self.__class__):
            return self.target_sector==target.target_sector and self.handlers==target.handlers
        else:
            raise TypeError('Can not compare %s and %s'%(self.__class__,target.__class__))

    def project_state(self,phi):
        '''
        project phi into specific discrete symmtry space.

        Parameters:
            :phi: 1D array, the state vector.

        Return:
            1D array, the state after projection.
        '''
        target_sector=self.target_sector
        for symm in self.symms:
            handler=self.handlers[symm]
            phi=handler.project_state(phi,parity=target_sector[symm])
        return phi

    def get_projector(self):
        '''
        Get the specific projection operator.

        Parameters:
            :target_sector: dict, {parity type:sector}

        Return:
            matrix, the projection matrix.
        '''
        target_sector=self.target_sector
        if len(target_sector)==0 or len(self.symms)==0:
            return None
        pl=[]
        for symm in self.symms:
            pl.append(self.handlers[symm].get_projector(target_sector[symm]))
        return prod(pl)

    def project_op(self,op):
        '''
        project operator(e.g. hamiltonian) into specific discrete symmtry space.

        Parameters:
            :op: matrix, the operator.

        Return:
            matrix, the hamiltonian after projection.
        '''
        target_sector=self.target_sector
        for symm in self.symms:
            handler=self.handlers[symm]
            op=handler.project_op(op,parity=target_sector[symm])
        return op

    def update_handlers(self,OPL=None,OPR=None,n=None,useC=True):
        '''
        Update handlers using provided parameters.

        Parameters:
            :n:, integer, the number of particle for left-right blocks, used for C2 symmetry.
            :OPL/OPR: dict, the {name:matrix} tuple to store the operator information.
        '''
        self.useC=useC
        if self.has_symmetry('C'):
            if n is None:
                raise Exception('n is required for C symmetry.')
            else:
                self.handlers['C'].update(n=n)
        for symm in ['J','P']:
            if self.has_symmetry(symm):
                if not OPL.has_key(symm) or not OPR.has_key(symm):
                    raise Exception('Can not find key %s in data.'%symm)
                data=kron(OPL[symm],OPR[symm])
                self.handlers[symm].update(data=data)

    def has_symmetry(self,symm):
        '''
        Check if this handler cope with specific symmetry.

        Parameters:
            :symm: char, the specific symmetry.

        Return:
            bool
        '''
        res=self.target_sector.has_key(symm)
        if symm=='C' and not self.useC:
            res=False
        return res

    def check_op(self,op):
        '''
        Check whether an operator do obey these discrete symmetries.
        '''
        return all([self.handlers[symm].check_op(op) for symm in self.symms])

    def check_parity(self,phi,**kwargs):
        '''
        check the parity of a state.

        Parameters:
            :phi: 1D array, the state vector.

        Return:
            bool, the state is qualified or not.
        '''
        overlap=abs(phi.dot(self.project_state(phi).conj()))/norm(phi)**2
        return overlap

    def locate(self,phis):
        '''
        locate the desired state from multiple eigen states.

        Parameters:
            :phis: list, the eigen-states.

        Return:
            1D array, the index of states meeting requirements.
        '''
        return where([self.check_parity(phi)>0.45 for phi in phis])[0]
