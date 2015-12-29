'''
Utilities for Heisenberg model.
'''
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import pdb,time,copy

from tba.hgen import SpinSpaceConfig
from core.mpo import MPO,OpUnitI
from core.mpolib import opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy
from core.mps import MPS
from dmrg import DMRGEngine
from hexpand.spinhexpand import DMRGSpinHGen
from hexpand.evolutor import MaskedEvolutor,NullEvolutor
from vmps import VMPSEngine
from vmpsapp import VMPSApp

class HeisenbergModel(VMPSApp):
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

class HeisenbergModel2(VMPSApp):
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

    def test_vmps(self):
        '''
        Run vMPS for Heisenberg model.
        '''
        filename='mps_heisenberg_%s.dat'%(nsite)
        model=self.get_model()
        if append:
            mps=MPS.load(filename)
        else:
            #run dmrg to get the initial guess.
            hgen=DMRGSpinHGen(spaceconfig=SpinSpaceConfig([2,1]),evolutor=MaskedEvolutor(hndim=2))
            dmrgegn=DMRGEngine(hchain=model.H_serial,hgen=hgen,tol=0)
            dmrgegn.run_finite(endpoint=(1,'<-',0),maxN=40,tol=1e-12)
            #hgen=dmrgegn.query('r',nsite-1)
            mps=dmrgegn.get_mps(direction='<-')  #right normalized initial state
            mps.save(filename)

        #run vmps
        vegn=VMPSEngine(H=model.H,k0=mps)
        vegn.run()

    def test_dmrg_finite(self):
        '''
        Run iDMFT for heisenberg model.
        '''
        model=self.get_model(10,1)
        hgen1=DMRGSpinHGen(spaceconfig=SpinSpaceConfig([2,1]),evolutor=NullEvolutor(hndim=2))
        hgen2=DMRGSpinHGen(spaceconfig=SpinSpaceConfig([2,1]),evolutor=MaskedEvolutor(hndim=2))
        dmrgegn=DMRGEngine(hchain=model.H_serial,hgen=hgen1,tol=0)
        EG1=dmrgegn.direct_solve()
        dmrgegn=DMRGEngine(hchain=model.H_serial,hgen=hgen2,tol=0)
        EG2=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=[10,20,30,40,40],tol=0)[-1]
        assert_almost_equal(EG1,EG2,decimal=4)

    def test_dmrg_infinite(self):
        '''test for infinite dmrg.'''
        maxiter=100
        model=self.get_model(maxiter+2,1)
        hgen=DMRGSpinHGen(spaceconfig=SpinSpaceConfig([2,1]),evolutor=MaskedEvolutor(hndim=2))
        dmrgegn=DMRGEngine(hchain=model.H_serial,hgen=hgen,tol=0)
        EG=dmrgegn.run_infinite(maxiter=maxiter,maxN=20,tol=0)[-1]
        assert_almost_equal(EG,0.25-log(2),decimal=2)

DMRGTest().test_dmrg_finite()
#DMRGTest().test_dmrg_infinite()
