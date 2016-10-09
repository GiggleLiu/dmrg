'''
TEBD related methods.
'''

from numpy import *
from scipy.linalg import expm,svd,norm
import pdb,time,copy

from rglib.mps import MPSBase,Tensor,OpCollection
from rglib.hexpand import RGHGen
from lanczos import get_H

__all__=['IVMPS','ITEBDEngine','entanglement_entropy']

class IVMPS(MPSBase):
    '''
    The infinite Vidal-form Matrix Product State.
    e.g. GL[0]-LL[0]-GL[1]-LL[1]-GL[2]-LL[2] ...

    Attributes:
        :hndim: int, the number of channels on each site.
        :nsite: int, the number of sites.
        :order: 1Darray, the order of the axes.
        :site_axis/llink_axis/rlink_axis: int, the specific axes for site, left link, right link.
        :npart: int, the number of parts(Readonly).
    '''
    def __init__(self,GL,LL,labels=['s','a']):
        llink_axis,site_axis,rlink_axis=self.llink_axis,self.site_axis,self.rlink_axis
        self.GL=[]
        for i,Gi in enumerate(GL):
            if not isinstance(Gi,Tensor):
                Gi=Tensor(Gi,labels=['']*3)
            Gi.labels[llink_axis]='%s_%s'%(labels[1],i)
            Gi.labels[rlink_axis]='%s_%s'%(labels[1],i+1)
            Gi.labels[site_axis]='%s_%s'%(labels[0],i)
            self.GL.append(Gi)
        self.LL=LL
        self.labels=labels

    def __copy__(self):
        return IVMPS([copy.copy(g) for g in self.GL],self.LL[:],self.labels[:])

    @property
    def hndim(self):
        '''The number of state in a single site.'''
        return self.GL[0].shape[self.site_axis]

    @property
    def nsite(self):
        '''Number of sites.'''
        return Inf

    @property
    def npart(self):
        '''The number of parts.'''
        return len(self.GL)

    def roll(self,x=1):
        '''
        Roll the order.

        Parameters:
            :x: int, the offset.
        
        Return:
            <IVMPS>,
        '''
        npart=self.npart
        GL2,LL2=[],[]
        for i in xrange(self.npart):
            GL2.append(copy.copy(self.GL[(i+x)%npart]))
            LL2.append(copy.copy(self.LL[(i+x)%npart]))
        mps2=IVMPS(GL=GL2,LL=LL2,labels=self.labels[:])
        return mps2

    def show(self,*args,**kwargs):
        '''Show this MPS graphically'''
        raise NotImplementedError()


class ITEBDEngine(object):
    '''
    Engine for Time Evolving Block Dicimation.

    Attributes:
        :status: list, the status [istep].
    '''
    def __init__(self,tol=1e-10):
        self.status=[0]
        self.tol=tol

    def reset(self):
        '''Reset all the run-time variables.'''
        self.status=[0]

    def update(self,ivmps,UL,maxN):
        '''
        Update the infinite VMPS object.

        Parameters:
            :ivmps: <IVMPS>, infinite VMPS.
            :UL: list, list of 4 leg tensor.
            :maxN: int, the maximum number of retained states.

        Return:
            <IVMPS>, new state.
        '''
        npart=ivmps.npart
        llink_axis,site_axis,rlink_axis=ivmps.llink_axis,ivmps.site_axis,ivmps.rlink_axis
        GL,LL=ivmps.GL,ivmps.LL
        for i in xrange(npart):
            #preparation
            i0,ia,ib=(i-1)%npart,i%npart,(i+1)%npart
            chi0,chi1,chi2=GL[ia].shape[llink_axis],GL[ia].shape[rlink_axis],GL[ib].shape[rlink_axis]
            d1,d2=GL[ia].shape[site_axis],GL[ib].shape[site_axis]
            Ui=UL[ia]

            #first evaluate theta
            theta=GL[ia].mul_axis(LL[i0],axis=llink_axis).mul_axis(LL[ia],axis=rlink_axis)
            theta2=GL[ib].mul_axis(LL[ib],rlink_axis)
            theta2.labels[llink_axis]=theta.labels[rlink_axis]='a_0'
            theta.labels[llink_axis]='a_1'
            theta2.labels[rlink_axis]='a_1\''
            theta=theta*theta2

            #apply Ui on ivmps, Ui is a 4-leg tensor
            Ui.labels=['Us1','Us2',GL[ia].labels[site_axis],GL[ib].labels[site_axis]]
            theta=(theta*Ui).chorder([0,2,3,1])
            theta=theta.reshape([chi0*d1,-1])

            #SVD, and truncate.
            U,S,V=svd(theta)
            chi=min(sum(S>self.tol),maxN)
            U=Tensor(U[:,:chi].reshape([chi0,d1,chi]),labels=GL[ia].labels)
            V=Tensor(V[:chi,:].reshape([chi,d2,chi2]),labels=GL[ib].labels)
            GL[ia],GL[ib]=U.mul_axis(1./LL[i0],axis=llink_axis),V.mul_axis(1./LL[ib],axis=rlink_axis)
            LL[ia]=S[:chi]; LL[ia]/=norm(LL[ia])
        self.status[0]+=1
        return ivmps

    def run(self,hs,ivmps,spaceconfig,maxN=20,dt=0.01,Nt=800):
        '''
        Get the ground state and ground state energy for given Hamiltonian.

        Parameters:
            :hs: list/<OpCollection>, a list of site-hamiltonians, or the site-hamiltonian.
            :ivmps: <IVMPS>, the initial state.
            :maxN: int, the maximum kept states.
            :dt/Nt: float/int, the time step and the number of time slices.
            :spaceconfig: <SpaceConfig>, single site hilbert space config.

        Return:
            (Emin, Vmin)
        '''
        npart=ivmps.npart
        site_axis=ivmps.site_axis
        GL=ivmps.GL

        #first, we get U-matrices, used as time evolusion matrices for states.
        if not isinstance(hs,(list,tuple)):
            hs=[hs]
        else:
            assert(len(hs)==npart)
        UL=[]
        for i,hi in enumerate(hs):
            ia,ib=i%npart,(i+1)%npart
            d1,d2=GL[ia].shape[site_axis],GL[ib].shape[site_axis]
            H=get_H(RGHGen(spaceconfig=spaceconfig,H=hi,evolutor_type='null')).todense()
            U=expm(-dt*get_H(RGHGen(spaceconfig=spaceconfig,H=hi,evolutor_type='null')).todense())
            U=Tensor(U.reshape([d1,d2,d1,d2]),labels=['Us1','Us2',GL[ia].labels[site_axis],GL[ib].labels[site_axis]])
            UL.append(U)
        if len(UL)==1:
            UL=UL*npart

        #optimize ivmps
        for i in xrange(Nt-1):
            ivmps=self.update(ivmps,UL,maxN=maxN)

        return ivmps

def random_IVMPS(d,N=20):
    '''
    Generate a random infinite VMPS.

    Parameters:
        :d: int, the dimension of physical site.
        :N: int, the number of retained states.

    Return:
        <IVMPS>,
    '''
    GL=[]
    for i in xrange(npart):
        Gi=random.rand(N,d,N)
        GL.append(Gi)
    LL=[]
    for i in xrange(npart):
        LL.append(ones([N]))
    ivmps=IVMPS(GL=GL,LL=LL)
    return ivmps

def entanglement_entropy(ivmps):
    '''
    Returns the half chain entanglement entropy

    Parameters:
        :ivmps: <IVMPS>,

    Return:
        int,
    '''
    l_list=ivmps.LL
    S=[]
    for i_bond in range(2):
        x=l_list[i_bond][l_list[i_bond]>10**(-20)]**2
        S.append(-inner(log(x),x))
    return S

