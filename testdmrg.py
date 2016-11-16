'''
Utilities for Heisenberg model.
'''
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
import pdb,time,copy

from tba.hgen import SpinSpaceConfig
from rglib.mps import WL2OPC,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS
from rglib.hexpand import ExpandGenerator
from rglib.hexpand import MaskedEvolutor,NullEvolutor,Evolutor
from dmrg import DMRGEngine
from lanczos import get_H,get_H_bm

class HeisenbergModel(object):
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
        self.H_serial=WL2OPC(WL)

class HeisenbergModel2(object):
    '''
    Heisenberg model application for vMPS.

    The Hamiltonian is: sum_i J/2*(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + Jz*S_i^zS_{i+1}^z -h*S_i^z

    Construct
    -----------------
    HeisenbergModel(J,Jz,h)

    Attributes
    ------------------
    J/Jz/J2/Jz2:
        Exchange interaction at xy direction and z direction, and the same parameter for second nearest hopping term.
    h:
        The strength of weiss field.
    *see VMPSApp for more attributes.*
    '''
    def __init__(self,J,Jz,h,nsite,J2=0,J2z=None):
        self.spaceconfig=SpinSpaceConfig([2,1])
        I=OpUnitI(hndim=2)
        scfg=SpinSpaceConfig([2,1])
        Sx=opunit_Sx(spaceconfig=scfg)
        Sy=opunit_Sy(spaceconfig=scfg)
        Sz=opunit_Sz(spaceconfig=scfg)

        #with second nearest hopping terms.
        if J2==0:
            self.H_serial2=None
            return
        elif J2z==None:
            J2z=J2
        SL=[]
        for i in xrange(nsite):
            Sxi,Syi,Szi=copy.copy(Sx),copy.copy(Sy),copy.copy(Sz)
            Sxi.siteindex=i
            Syi.siteindex=i
            Szi.siteindex=i
            SL.append(array([Sxi,Syi,sqrt(J2z/J2)*Szi]))
        ops=[]
        for i in xrange(nsite-1):
            ops.append(J*SL[i].dot(SL[i+1]))
            if i<nsite-2 and J2z!=0:
                ops.append(J2*SL[i].dot(SL[i+2]))

        mpc=sum(ops)
        self.H_serial=mpc

class DMRGTest():
    '''
    Tests for dmrg and dmft.
    '''
    def __init__(self):
        pass

    def get_model(self,nsite,which):
        '''get a n-site model.'''
        J=1.
        Jz=1.
        J2=0.2
        J2z=0.2
        h=0
        if which==1:
            model=HeisenbergModel(J=J,Jz=Jz,h=h,nsite=nsite)
        else:
            model=HeisenbergModel2(J=J,Jz=Jz,J2=J2,J2z=J2z,h=h,nsite=nsite)
        return model

    def test_dmrg_finite(self):
        '''
        Run iDMFT for heisenberg model.
        '''
        nsite=10
        model=self.get_model(nsite,1)
        hgen1=ExpandGenerator(spaceconfig=SpinSpaceConfig([2,1]),H=model.H_serial,evolutor_type='null')
        hgen2=ExpandGenerator(spaceconfig=SpinSpaceConfig([2,1]),H=model.H_serial,evolutor_type='masked')
        H=get_H(hgen1)
        EG1=eigsh(H,k=1,which='SA')[0]
        dmrgegn=DMRGEngine(hgen=hgen2,tol=0,reflect=True)
        dmrgegn.use_U1_symmetry('M',target_block=zeros(1))
        EG2=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=[10,20,40,40,40],tol=0)[0]
        assert_almost_equal(EG1,EG2,decimal=4)

    def test_dmrg_infinite(self):
        '''test for infinite dmrg.'''
        maxiter=100
        model=self.get_model(maxiter+2,1)
        hgen=ExpandGenerator(spaceconfig=SpinSpaceConfig([2,1]),H=model.H_serial,evolutor_type='masked')
        dmrgegn=DMRGEngine(hgen=hgen,tol=0,reflect=True,iprint=10)
        dmrgegn.use_U1_symmetry('M',target_block=zeros(1))
        EG=dmrgegn.run_infinite(maxiter=maxiter,maxN=20,tol=0)[0]
        assert_almost_equal(EG,0.25-log(2),decimal=2)

    def test_lanczos(self):
        '''test for directly construct and solve the ground state energy.'''
        model=self.get_model(10,1)
        hgen1=ExpandGenerator(spaceconfig=SpinSpaceConfig([2,1]),H=model.H_serial,evolutor_type='null')
        hgen2=ExpandGenerator(spaceconfig=SpinSpaceConfig([2,1]),H=model.H_serial,evolutor_type='normal')
        dmrgegn=DMRGEngine(hgen=hgen1,tol=0,iprint=10)
        H=get_H(hgen=hgen1)
        H2,bm2=get_H_bm(hgen=hgen2,bstr='M')
        Emin=eigsh(H,k=1)[0]
        Emin2=eigsh(bm2.extract_block(H2,(zeros(1),zeros(1)),uselabel=True),k=1)[0]
        print 'The Ground State Energy is %s, tolerence %s.'%(Emin,Emin-Emin2)
        assert_almost_equal(Emin,Emin2)

DMRGTest().test_dmrg_finite()
DMRGTest().test_lanczos()
DMRGTest().test_dmrg_infinite()
