'''
Discrete symmetries.
'''

from numpy import *

from rglib.mps import OpUnit

__all__=['DiscSymm','PHSymm','FlipSymm','C2Symm']

class DiscSymm(object):
    '''
    Discrete symmetrix class.

    Attributes:
        :name: str, the name of this symmetrc.
        :val: number, the operation matrix of this symmetrc.
    '''
    def __init__(self,name,val=None):
        self.name=name
        self.val=val

    def act_on_state(self,phi):
        '''
        Act on the specific state.

        Parameters:
            :phi: 1D array, the state vector.

        Return:
            1D array, the state after parity action.
        '''
        return self.val.dot(phi)
        #raise NotImplementedError()

    def project_state(self,phi,parity,**kwargs):
        '''
        symmetrize a state.

        Parameters:
            :phi: 1D array, the state vector.
            :parity: 1/-1, the specific parity.

        Return:
            1D array, the state after projection.
        '''
        assert(parity==1 or parity==-1)
        phi2=self.act_on_state(phi,**kwargs)
        return (phi+parity*phi2)/2.

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

class PHSymm(DiscSymm):
    '''
    Particle hole symmetry.
    '''
    def __init__(self,val=None):
        super(PHSymm,self).__init__('ph',val)

    def J(self,i):
        '''
        Get the transformation opunit at specific site.

        Parameters:
            :i: int, the site index.
        '''
        data=zeros([4,4])
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
    def __init__(self,val=None):
        super(FlipSymm,self).__init__('sf',val)

    def P(self,i):
        '''
        Get the transformation opunit at specific site.

        Parameters:
            :i: int, the site index.
        '''
        data=zeros([4,4])
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
    def __init__(self,val=None):
        super(C2Symm,self).__init__('c2',val)

    def act_on_state(self,phi,nl,nr,**kwargs):
        '''
        Assuming B.B. Hilbert space configuration.

        Parameters:
            :phi: 1D array, the state.
            :nl/nr: int, the number of electrons in left and right blocks.

        Return:
            1D array, the new state after C2 action.
        '''
        N=sqrt(len(phi))
        phi=phi.reshape([N,N])
        factor=(-1)**((nl[:,newaxis]*nr)%2)
        phi2=(phi*factor).T.ravel()
        return phi2
