'''
Variational Matrix Product State.
'''

from numpy import *
from scipy.linalg import svd,eigh,norm
from matplotlib.pyplot import *
import time,pdb

from rglib.mps import Tensor,TNet
from tba.lattice import Structure,Triangular_Lattice

__all__=['SRGEngine']

ZERO_REF=1e-12

def construct_tnet(N,geometry='tri'):
    '''
    Construct Tensor Network.

    Parameters:
        :N: int, the size of geometry.
        :geometry: str, the shape of lattice.

    Return:
        <TNet>,
    '''
    if geometry=='tri':
        st=Triangular_Lattice(N=(N,N))
        a=(st.a[0]+st.a[1])/2.
        a/=norm(a)
        na=norm(st.a[0])
        sites=filter(lambda v: v.dot(a)/na<sqrt(3.)/2.*N,st.sites)
        st=Structure(sites)
        ion()
        scatter(st.sites[:,0],st.sites[:,1])
        pdb.set_trace()

construct_tnet(10)

class SRGEngine(object):
    '''
    Variational MPS Engine.

    Attributes:
        :geometry: str, the geometry of tensor network.
    '''
    def __init__(self,geometry):
        pass

    def renormalize(TNet):
        pass
