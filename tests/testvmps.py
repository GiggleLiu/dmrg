from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
import pdb,time,copy,sys
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig
from pymps import WL2MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,\
        product_state,random_product_state,WL2OPC,check_validity_op
from rglib.hexpand import ExpandGenerator,geth_expand
from rglib.hexpand import MaskedEvolutor,NullEvolutor,Evolutor
from dmrg import DMRGEngine
from blockmatrix import SimpleBMG

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
    def __init__(self,J,Jz,h,nsite,nspin=3):
        self.spaceconfig=SpinSpaceConfig([1,nspin])
        scfg=self.spaceconfig
        I=OpUnitI(hndim=scfg.hndim)
        Sx=opunit_Sx(spaceconfig=scfg)
        Sy=opunit_Sy(spaceconfig=scfg)
        Sz=opunit_Sz(spaceconfig=scfg)
        Sp=opunit_Sp(spaceconfig=scfg)
        Sm=opunit_Sm(spaceconfig=scfg)
        wi=zeros((5,5),dtype='O')
        #wi[:,0],wi[4,1:]=(I,Sp,Sm,Sz,-h*Sz),(J/2.*Sm,J/2.*Sp,Jz*Sz,I)
        wi[:,0],wi[4,1:]=(I,Sx,Sy,Sz,-h*Sz),(J*Sx,J*Sy,Jz*Sz,I)
        WL=[copy.deepcopy(wi) for i in xrange(nsite)]
        WL[0]=WL[0][4:]
        WL[-1]=WL[-1][:,:1]
        self.H=WL2MPO(WL)
        mpc=WL2OPC(WL)
        self.H_serial=mpc


class TestVMPS(object):
    def get_model(self,nsite,nspin=3):
        '''get a n-site model.'''
        J=1.
        Jz=1.
        J2=0.2
        J2z=0.2
        h=0
        model=HeisenbergModel(J=J,Jz=Jz,h=h,nsite=nsite,nspin=nspin)
        return model

    def dmrgrun(self,model):
        '''Get the result from DMRG'''
        #run dmrg to get the initial guess.
        hgen=ExpandGenerator(spaceconfig=model.spaceconfig,H=model.H_serial,evolutor_type='masked')
        dmrgegn=DMRGEngine(hgen=hgen,tol=0,reflect=True)
        dmrgegn.use_U1_symmetry('M',target_block=zeros(1))
        EG,mps=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=30,tol=1e-12)
        return EG,mps

    def test_vmps(self):
        '''
        Run vMPS for Heisenberg model.
        '''
        nsite=10
        nspin=2
        model=self.get_model(nsite,nspin=nspin)
        #EG,mps=self.dmrgrun(model)

        #run vmps
        #generate a random mps as initial vector
        bmg=SimpleBMG(spaceconfig=model.spaceconfig,qstring='M')
        #k0=product_state(config=random.randint(0,2,nsite),hndim=2)
        if nspin==2:
            k0=product_state(config=repeat([0,1],nsite/2),hndim=model.spaceconfig.hndim,bmg=bmg)
        else:
            k0=product_state(config=repeat([0,2],nsite/2),hndim=model.spaceconfig.hndim,bmg=bmg)

        #setting up the engine
        vegn=VMPSEngine(H=model.H.use_bm(bmg),k0=k0,eigen_solver='JD')
        #check the label setting is working properly
        assert_(all([ai.shape==(ai.labels[0].bm.N,ai.labels[1].bm.N,ai.labels[2].bm.N) for ai in vegn.ket.ML]))
        assert_(all([ai.shape==(ai.labels[0].bm.N,ai.labels[1].bm.N,ai.labels[2].bm.N,ai.labels[3].bm.N) for ai in vegn.H.OL]))
        #warm up
        vegn.warmup(10)
        vegn.eigen_solver='LC'
        vegn.run(5,maxN=[40,100,200,200,200,200,200],which='SA')
        pdb.set_trace()
        vegn.run(5,maxN=80,which='SA')
        pdb.set_trace()

    def test_vmps2(self):
        '''
        Run vMPS for Heisenberg model.
        '''
        nsite=20
        model=self.get_model(nsite)
        #EG,mps=self.dmrgrun(model)

        #run vmps
        #generate a random mps as initial vector
        #k0=product_state(config=random.randint(0,2,nsite),hndim=3)
        k0=product_state(config=repeat([0,1],nsite/2),hndim=model.spaceconfig.hndim)

        #setting up the engine
        vegn=VMPSEngine(H=model.H,k0=k0,eigen_solver='LC',nsite_update=2)
        #check the label setting is working properly
        vegn.run(4,maxN=40,which='SA')
        pdb.set_trace()
        vegn.run(4,maxN=40,which='SA')
        pdb.set_trace()

    def test_iterator(self):
        nsite=4
        model=self.get_model(nsite)
        k0=product_state(config=repeat([0,1],nsite/2),hndim=model.spaceconfig.hndim)

        vegn=VMPSEngine(H=model.H,k0=k0,eigen_solver='LC')
        start=(1,'->',2)
        stop=(3,'<-',1)
        print 'Testing iterator start = %s, stop= %s'%(start,stop)
        iterator=vegn._get_iterator(start=start,stop=stop)
        order=[(1,'->',2),(1,'<-',1),(1,'<-',0),
                (2,'->',1),(2,'->',2),(2,'<-',1),(2,'<-',0),
                (3,'->',1),(3,'->',2),(3,'<-',1),
                ]
        for od,it in zip(order,iterator):
            assert_(od==it)

if __name__=='__main__':
    TestVMPS().test_vmps()
    #TestVMPS().test_iterator()
