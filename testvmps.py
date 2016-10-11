from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
import pdb,time,copy

from tba.hgen import SpinSpaceConfig
from rglib.mps import MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,product_state,random_product_state
from rglib.hexpand import RGHGen
from rglib.hexpand import MaskedEvolutor,NullEvolutor,Evolutor
from dmrg import DMRGEngine
from lanczos import get_H,get_H_bm

from vmps import VMPSEngine

class HeisenbergModel():
    '''
    Heisenberg model application for vMPS.

    The Hamiltonian is: sum_i J/2*(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + Jz*S_i^zS_{i+1}^z -h*S_i^z

    Construct
    -----------------
    HeisenbergModel(J,Jz,h)

    Attributes
    ------------------
    J/Jz:
        Exchange interaction at xy direction and z direction.
    h:
        The strength of weiss field.
    *see VMPSApp for more attributes.*
    '''
    def __init__(self,J,Jz,h,nsite):
        self.spaceconfig=SpinSpaceConfig([2,1])
        I=OpUnitI(hndim=2)
        scfg=SpinSpaceConfig([2,1])
        Sz=opunit_Sz(spaceconfig=scfg)
        Sp=opunit_Sp(spaceconfig=scfg)
        Sm=opunit_Sm(spaceconfig=scfg)
        wi=zeros((5,5),dtype='O')
        wi[:,0],wi[4,1:]=(I,Sp,Sm,Sz,-h*Sz),(J/2.*Sm,J/2.*Sp,Jz*Sz,I)
        WL=[copy.deepcopy(wi) for i in xrange(nsite)]
        WL[0]=WL[0][4:]
        WL[-1]=WL[-1][:,:1]
        self.H=MPO(WL)
        mpc=self.H.serialize()
        mpc.compactify()
        self.H_serial=mpc


class TestVMPS(object):
    def get_model(self,nsite):
        '''get a n-site model.'''
        J=1.
        Jz=1.
        J2=0.2
        J2z=0.2
        h=0
        model=HeisenbergModel(J=J,Jz=Jz,h=h,nsite=nsite)
        return model

    def dmrgrun(self,model):
        '''Get the result from DMRG'''
        #run dmrg to get the initial guess.
        hgen=RGHGen(spaceconfig=SpinSpaceConfig([2,1]),H=model.H_serial,evolutor_type='masked')
        dmrgegn=DMRGEngine(hgen=hgen,tol=0,reflect=True)
        dmrgegn.use_U1_symmetry('M',target_block=zeros(1))
        EG,mps=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=40,tol=1e-12)
        return EG,mps

    def test_vmps(self):
        '''
        Run vMPS for Heisenberg model.
        '''
        nsite=10
        model=self.get_model(nsite)
        #EG,mps=self.dmrgrun(model)

        #run vmps
        #generate a random mps as initial vector
        #k0=product_state(config=random.randint(0,2,nsite),hndim=2)
        k0=random_product_state(nsite=nsite,hndim=2)
        vegn=VMPSEngine(H=model.H,k0=k0)
        vegn.run()

if __name__=='__main__':
    TestVMPS().test_vmps()
