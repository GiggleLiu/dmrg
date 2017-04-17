from numpy import *
from numpy.linalg import norm,svd
from copy import deepcopy
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
import sys,pdb,time
from os import path
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig
from pymps.mpolib import *
from pymps.mpo import *
from pymps.mpslib import *
from contractor import Contractor

class TestCon(object):
    '''
    Contract for addition of two <MPS> instances.
    '''
    def __init__(self):
        nsite=6   #number of sites
        hndim=2
        l=0
        vec=random.random(hndim**nsite)   #a random state in form of 1D array.
        vec2=random.random(hndim**nsite)  #a random state in form of 1D array.

        mps=state2MPS(vec,sitedim=hndim,l=l,method='svd')     #parse the state into a <MPS> instance.
        mps2=state2MPS(vec2,sitedim=hndim,l=l,method='svd')     #parse the state into a <MPS> instance.

        j1,j2=0.5,0.2
        scfg=SpinSpaceConfig([1,2])
        I=OpUnitI(hndim=hndim)
        Sz=opunit_Sz(spaceconfig=scfg)
        Sp=opunit_Sp(spaceconfig=scfg)
        Sm=opunit_Sm(spaceconfig=scfg)
        wi=zeros((4,4),dtype='O')
        wi[0,0],wi[1,0],wi[2,1],wi[3,1:]=I,Sz,I,(j1*Sz,j2*Sz,I)
        WL=[deepcopy(wi) for i in xrange(nsite)]
        WL[0]=WL[0][3:4]
        WL[-1]=WL[-1][:,:1]
        mpo=WL2MPO(WL)

        self.mps,self.mps2=mps,mps2
        self.mpo=mpo
        self.vec,self.vec2=vec,vec2
        self.spaceconfig=scfg
     
    def test_braOket(self):
        print 'Testing contraction of mpses.'
        con=Contractor(self.mpo,self.mps,bra_bond_str='c')
        con.initialize_env()
        S2=con.RPART[-1]*self.mps.S**2
        l0=3
        self.mps>>l0
        S0=con.evaluate()
        H=self.mpo.H
        v=self.mps.state
        S1=v.conj().dot(H.dot(v))
        self.mps>>self.mps.nsite-l0
        for i in xrange(self.mps.nsite):
            con.lupdate_env(i+1)
        S3=con.LPART[self.mps.nsite]*self.mps.S**2
        con.canomove(-3)
        print con
        con.keep_only(2,4)
        print con
        pdb.set_trace()
        S4=con.evaluate()
        assert_almost_equal(S0,S1)
        assert_almost_equal(S0,S2)
        assert_almost_equal(S0,S3)

    def test_all(self):
        self.test_braOket()

if __name__=='__main__':
    TestCon().test_all()
