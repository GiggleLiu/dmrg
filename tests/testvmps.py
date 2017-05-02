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
from toymodel import HeisenbergModel,HeisenbergModel2D


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

    def get_model2d(self,N,nspin=3):
        '''get a n-site model.'''
        J=1.
        Jz=1.
        J2=0.2
        J2z=0.2
        h=0
        model=HeisenbergModel2D(J=J,Jz=Jz,h=h,N1=N,N2=N,nspin=nspin)
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
        pdb.set_trace()
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

        print 'Testing 2-site iterator.'
        nsite=2
        model=self.get_model(nsite)
        k0=product_state(config=repeat([0,1],nsite/2),hndim=model.spaceconfig.hndim)

        vegn=VMPSEngine(H=model.H,k0=k0,eigen_solver='LC')
        start=(1,'->',0)
        stop=(3,'->',0)
        order=[(1,'->',0),(2,'->',0),(3,'->',0)]
        iterator=vegn._get_iterator(start=start,stop=stop)
        for od,it in zip(order,iterator):
            assert_(od==it)

    def test_generative(self,trunc_mps):
        '''
        Run vMPS for Heisenberg model.
        '''
        nsite=4
        nspin=2
        model=self.get_model(nsite,nspin=nspin)
        model2=self.get_model(nsite*2,nspin=nspin)
        #EG,mps=self.dmrgrun(model)

        #run vmps
        #generate a random mps as initial vector
        bmg=SimpleBMG(spaceconfig=model.spaceconfig,qstring='M')
        #k0=product_state(config=random.randint(0,2,nsite),hndim=2)
        k0=product_state(config=repeat([0,nspin-1],nsite/2),hndim=model.spaceconfig.hndim,bmg=bmg)

        #setting up the engine
        vegn=VMPSEngine(H=model.H.use_bm(bmg),k0=k0,eigen_solver='JD',iprint=1)
        niter_inner=4 if nsite>2 else 1
        vegn.run(niter_inner,maxN=50,which='SA')
        #warm up
        vegn.generative_run(HP=model2.H.OL[nsite/2:nsite*3/2],ngen=100,niter_inner=niter_inner,maxN=50,trunc_mps=trunc_mps,which='SA')
        pdb.set_trace()

    def test_2d(self):
        '''
        Run vMPS for Heisenberg model.
        '''
        N1,N2=4,4
        nsite=N1*N2
        nspin=2
        model=self.get_model2d(N=N1,nspin=nspin)
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
        #mpo1=copy.deepcopy(model.H).compress(kernel='svd',tol=1e-8)[0]
        mpo=copy.deepcopy(model.H).compress(kernel='dpl',tol=1e-8)[0]
        #mpo3=copy.deepcopy(model.H).compress(kernel='ldu',tol=1e-8)[0]
        vegn=VMPSEngine(H=mpo.use_bm(bmg),k0=k0,eigen_solver='JD')
        #check the label setting is working properly
        assert_(all([ai.shape==(ai.labels[0].bm.N,ai.labels[1].bm.N,ai.labels[2].bm.N) for ai in vegn.ket.ML]))
        assert_(all([ai.shape==(ai.labels[0].bm.N,ai.labels[1].bm.N,ai.labels[2].bm.N,ai.labels[3].bm.N) for ai in vegn.H.OL]))
        #warm up
        vegn.warmup(10)
        vegn.run(5,maxN=[40,100,150,150,150,150,150],which='SA')
        pdb.set_trace()
        vegn.run(5,maxN=80,which='SA')
        pdb.set_trace()

if __name__=='__main__':
    #TestVMPS().test_2d()
    TestVMPS().test_vmps()
    #TestVMPS().test_iterator()
    #TestVMPS().test_generative(trunc_mps=True)
