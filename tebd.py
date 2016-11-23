'''
TEBD related methods.
'''

from numpy import *
from scipy.linalg import expm,svd,norm
import pdb,time,copy

from rglib.mps import MPSBase,Tensor,OpCollection,TNet,IVMPS,Link
from rglib.hexpand import geth_expand

__all__=['ITEBDEngine','entanglement_entropy']

class ITEBDEngine(object):
    '''
    Engine for Time Evolving Block Dicimation.

    Attributes:
        :links: hs, the hamiltonians defined on links.
    '''
    def __init__(self,hs,tol=1e-10):
        self.hs=hs
        self.tol=tol

    def evolve_single_step(self,ivmps,U,ilink,maxN):
        '''
        Update the infinite VMPS object.

        Parameters:
            :ivmps: <IVMPS>, infinite VMPS.
            :U: 4d array, unitary matrix.
            :ilink: int, optimize on i-th link.
            :maxN: int, the maximum number of retained states.

        Return:
            <IVMPS>, new state.
        '''
        site_axis=0
        link,legs=ivmps.LL[ilink],list(ivmps.legs)
        ta,la=ivmps.lid2tid(legs.index(link.labels[0]))  #la-th leg of tensor ta
        tb,lb=ivmps.lid2tid(legs.index(link.labels[1]))

        #first evaluate theta
        A=ivmps.attach_links(ivmps.tensors[ta],exception=link.labels)
        B=ivmps.attach_links(ivmps.tensors[tb],exception=link.labels)
        alabels=A.labels[:]; alabels[la]=B.labels[lb]
        theta=A.make_copy(labels=alabels,copydata=False)*B #labels -> A.labels(without la)+B.labels(without lb)

        #apply Ui on ivmps, Ui is a 4-leg tensor
        Ui=Tensor(U,labels=['Us1','Us2',A.labels[site_axis],B.labels[site_axis]])
        order=range(ndim(theta)); order.insert(ndim(A)-1,order.pop(1))
        theta0=(Ui*theta).chorder(order)
        theta=theta0.reshape([A.size/A.shape[la],B.size/B.shape[lb]])

        #SVD, and truncate.
        U,S,V=svd(theta)
        chi=min(sum(S>self.tol),maxN)
        order1,order2=range(3),range(3)
        order1.insert(la,order1.pop(-1))
        order2.insert(lb,order2.pop(0))
        U=ivmps.detach_links(Tensor(U[:,:chi].reshape(theta0.shape[:ndim(theta0)/2]+(chi,)).transpose(order1),labels=A.labels),exception=link.labels)
        V=ivmps.detach_links(Tensor(V[:chi,:].reshape((chi,)+theta0.shape[ndim(theta0)/2:]).transpose(order2),labels=B.labels),exception=link.labels)
        link.S=S[:chi]; link.S/=norm(link.S)
        ivmps.tensors[ta],ivmps.tensors[tb]=U,V
        return ivmps

    def run(self,ivmps,maxN=20,dt=0.01,Nt=800):
        '''
        Get the ground state and ground state energy for given Hamiltonian.

        Parameters:
            :ivmps: <IVMPS>, the initial state.
            :maxN: int, the maximum kept states.
            :dt/Nt: float/int, the time step and the number of time slices.

        Return:
            (Emin, Vmin)
        '''
        if len(self.hs)!=len(ivmps.LL): raise ValueError()
        #get time evolving matrices.
        UL=[]
        for i,hi in enumerate(self.hs):
            U=expm(-dt*hi.H(nsite=2))
            UL.append(U.reshape([ivmps.hndim]*4))

        #optimize ivmps
        for i in xrange(Nt-1):
            for ilink in xrange(len(UL)):
                ivmps=self.evolve_single_step(ivmps,UL[ilink],ilink,maxN=maxN)
                print ivmps.LL[0]
                pdb.set_trace()
        return ivmps

def random_ivmps_chain(hndim,N=5,labels=['s','a','b']):
    '''
    Generate a random infinite VMPS on a A,B chain.

    Parameters:
        :hndim: int, the dimension of physical site.
        :N: int, the number of retained states.
        :labels: list, the label for (site, left bond, right bond).

    Return:
        <IVMPS>,
    '''
    tensors,LL=[],[]
    for i in xrange(2):
        Gi=tensor(random.random([hndim,N,N]),labels=['%s_%s'%(labels[j],i) for j in xrange(3)])
        tensors.append(Gi)
    for i in xrange(2):
        li=sort(random.random([N]))[::-1]
        li/=norm(li)
        LL.append(Link(li,labels=['%s_%s'%(labels[i],j) for j in [0,1]]))
    ivmps=IVMPS(tensors=tensors,LL=LL)
    return ivmps

def random_ivmps_honeycomb(hndim,N=5,labels=['s','a','b','d']):
    '''
    Generate a random infinite VMPS on a honeycomb lattice.

    Parameters:
        :hndim: int, the dimension of physical site.
        :N: int, the number of retained states.
        :labels: list, the label for (site, left bond, right bond).

    Return:
        <IVMPS>,
    '''
    tensors,LL=[],[]
    for i in xrange(2):
        Gi=tensor(random.random([hndim,N,N,N]),labels=['%s_%s'%(labels[j],i) for j in xrange(4)])
        tensors.append(Gi)
    for i in xrange(3):
        li=sort(random.random([N]))[::-1]
        li/=norm(li)
        LL.append(Link(li,labels=['%s_%s'%(labels[i],j) for j in [0,1]]))
    ivmps=IVMPS(tensors=tensors,LL=LL)
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
    SL=[]
    for i_bond in range(2):
        S=l_list[i_bond].S
        x=S[S>10**(-20)]**2
        SL.append(-inner(log(x),x))
    return SL

