'''
Test for TEBD utilities.
'''
from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
from scipy import integrate
import pdb,time,copy,sys
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig
from rglib.mps import MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,Tensor,opunit_S,get_expect_ivmps,Link,IVMPS
from rglib.hexpand import ExpandGenerator
from rglib.hexpand import MaskedEvolutor,NullEvolutor,Evolutor
from tebd import *

class TestTEBD(object):
    def get_Ising(self,h,J=1.):
        '''Get the hamiltonian.'''
        #the hamiltonian for Ising model
        Sz=opunit_Sz(spaceconfig=SpinSpaceConfig([1,2]))
        Sx=opunit_Sx(spaceconfig=SpinSpaceConfig([1,2]))
        H=-J*Sz.as_site(0)*Sz.as_site(1)+h*Sx.as_site(0)
        return H

    def get_Haldane(self,D,J=1.):
        '''Get the hamiltonian.'''
        #the hamiltonian for Ising model
        Sx=opunit_Sx(spaceconfig=SpinSpaceConfig([1,3]))
        Sy=opunit_Sy(spaceconfig=SpinSpaceConfig([1,3]))
        Sz=opunit_Sz(spaceconfig=SpinSpaceConfig([1,3]))
        H=J*(Sz.as_site(0)*Sz.as_site(1)+Sx.as_site(0)*Sx.as_site(1)+Sy.as_site(0)*Sy.as_site(1))+D*Sz.as_site(0)*Sz.as_site(0)
        return H

    def test_Haldane(self):
        '''Solve model hamiltonians'''
        egn=ITEBDEngine(tol=1e-10)
        npart=2
        Dlist=arange(0,-0.6,-0.05)
        spaceconfig=SpinSpaceConfig([1,3])
        #the initial state
        GL=[]
        for i in xrange(npart):
            Gi=(0.5-random.rand(1,3,1))/100.
            Gi[0,2*(i%2),0]=1;
            GL.append(Gi)
        LL=[]
        for i in xrange(npart):
            LL.append(ones([1]))

        ivmps0=IVMPS(GL=GL,LL=LL)
        mpsl,HL,EEL=[],[],[]
        for D in Dlist:
            H=self.get_Haldane(J=1.,D=D)
            mps=egn.run(hs=H,ivmps=copy.copy(ivmps0),spaceconfig=spaceconfig,maxN=20)
            mpsl.append(mps)
            HL.append(H)
            print D,mps.LL[0][:4]

        SL,EL,ML1,ML2=[],[],[],[]
        for mps,H in zip(mpsl,HL):
            mps2=mps.roll(1)
            SL.append(mean(entanglement_entropy(mps)))
            print get_expect_ivmps(op=H,ket=mps),get_expect_ivmps(op=H,ket=mps2)
            EL.append(mean([get_expect_ivmps(op=H,ket=mps),get_expect_ivmps(op=H,ket=mps2)]))
            ML1.append(get_expect_ivmps(op=2*opunit_S(which='z',siteindex=0,spaceconfig=spaceconfig),ket=mps))
            ML2.append(get_expect_ivmps(op=2*opunit_S(which='z',siteindex=0,spaceconfig=spaceconfig),ket=mps2))

        ion()
        subplot(311)
        plot(Dlist,SL)
        subplot(312)
        plot(Dlist,array(EL))
        subplot(313)
        plot(Dlist,array(ML1))
        plot(Dlist,array(ML2))
        pdb.set_trace()


    def test_ising(self):
        '''Solve model hamiltonians'''
        npart=2
        hlist=arange(0.,2,0.1)
        spaceconfig=SpinSpaceConfig([1,2])
        #the initial state
        GL=[]
        for i in xrange(npart):
            Gi=Tensor(0.5-0.01*ones([spaceconfig.hndim,1,1]),labels=['s%s'%i,'a%s'%i,'b%s'%i])
            Gi[0,0,0]=1
            GL.append(Gi)
        GL[1].labels[1:]=GL[1].labels[1:][::-1]
        #print GL[0].labels
        #print GL[1].labels
        LL=[Link(['b0','b1'],ones([1])),Link(['a1','a0'],ones([1]))]

        mpsl,EEL,HL=[],[],[]
        for h in hlist:
            hb=self.get_Ising(h=2*h,J=4.)
            egn=ITEBDEngine(hs=[hb]*2,tol=1e-10)
            ivmps0=IVMPS(tensors=GL,LL=LL)
            mps=egn.run(ivmps=ivmps0,maxN=5)
            mpsl.append(mps)
            HL.append(hb)

            ################# Get the exact energies ######################
            f = lambda k,h : -2*sqrt(1+h**2-2*h*cos(k))/pi/2.
            E0_exact = integrate.quad(f, 0, pi, args=(h,))[0]     
            EEL.append(E0_exact)

        SL,EL,ML=[],[],[]
        for mps,H in zip(mpsl,HL):
            SL.append(entanglement_entropy(mps))
            #print get_expect_ivmps(op=H,ket=mps),get_expect_ivmps(op=H,ket=mps2)
            #EL.append(mean([get_expect_ivmps(op=H,ket=mps),get_expect_ivmps(op=H,ket=mps2)]))
            #ML.append(get_expect_ivmps(op=2*opunit_S(which='z',siteindex=0,spaceconfig=spaceconfig),ket=mps))

        ion()
        subplot(311)
        #plot(hlist,abs(array(ML)))
        subplot(312)
        plot(hlist,SL)
        subplot(313)
        #plot(hlist,abs(array(EL)-EEL))
        pdb.set_trace()

    def test_all(self):
        self.test_ising()
        self.test_Haldane()

TestTEBD().test_all()

