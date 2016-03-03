'''
Test program for DMRG fermionic solver.
'''

from numpy import *
from matplotlib.pyplot import *
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import scipy.sparse as sps
import time,pdb,warnings

from tba.hgen import RHGenerator,op_simple_hopping,op_U,SuperSpaceConfig,SpaceConfig,op_simple_onsite
from tba.lattice import Chain
from rglib.mps import op2collection
from rglib.hexpand import FermionHGen,NullEvolutor,MaskedEvolutor,Evolutor
from lanczos import get_H,get_H_bm
from blockmatrix import get_bmgen
from dmrg import *

class ChainN(object):
    '''This is a tight-binding model for a chain.'''
    def __init__(self,t,t2=0,U=0,mu=0.,occ=True,nsite=6):
        self.t,self.t2,self.U,self.mu=t,t2,U,mu
        self.occ=occ
        self.nsite=nsite

        #occupation representation will use <SuperSpaceConfig>, otherwise <SpaceConfig>.
        if self.occ:
            spaceconfig=SuperSpaceConfig([1,2,nsite,1])
        else:
            spaceconfig=SpaceConfig([1,2,nsite,1],kspace=False)
            if abs(U)>0: warnings.warn('U is ignored in non-occupation representation.')
        hgen=RHGenerator(spaceconfig=spaceconfig)

        #define the operator of the system
        hgen.register_params({
            't1':self.t,
            't2':self.t2,
            'U':self.U,
            '-mu':-self.mu,
            })

        #define a structure and initialize bonds.
        rlattice=Chain(N=nsite)
        hgen.uselattice(rlattice)

        b1s=rlattice.getbonds(1)  #the nearest neighbor
        b2s=rlattice.getbonds(2)  #the nearest neighbor

        #add the hopping term.
        op_t1=op_simple_hopping(label='hop1',spaceconfig=spaceconfig,bonds=b1s)
        hgen.register_operator(op_t1,param='t1')
        op_t2=op_simple_hopping(label='hop2',spaceconfig=spaceconfig,bonds=b2s)
        hgen.register_operator(op_t2,param='t2')
        op_n=op_simple_onsite(label='n',spaceconfig=spaceconfig)
        hgen.register_operator(op_n,param='-mu')

        #add the hubbard interaction term if it is in the occupation number representation.
        if self.occ:
            op_ninj=op_U(label='ninj',spaceconfig=spaceconfig)
            hgen.register_operator(op_ninj,param='U')

        self.hgen=hgen

class TestFH(object):
    '''
    Test for fermionic dmrg.
    '''
    def __init__(self):
        self.set_params()

    def set_params(self,U=0.,t=1.,mu=0.1,t2=0.,nsite=6):
        '''Set the parameters.'''
        self.t,self.U,self.t2,self.mu=t,U,t2,mu
        self.model_exact=ChainN(t=t,U=U,t2=t2,mu=mu,occ=False,nsite=nsite)
        self.model_occ=ChainN(t=t,U=U,t2=t2,mu=mu,occ=True,nsite=nsite)
        scfg=self.model_occ.hgen.spaceconfig
        spaceconfig=SuperSpaceConfig([scfg.nspin,1,scfg.norbit])
        self.expander=FermionHGen(spaceconfig=spaceconfig,evolutor=NullEvolutor(hndim=spaceconfig.hndim))
        self.expander2=FermionHGen(spaceconfig=spaceconfig,evolutor=Evolutor(hndim=spaceconfig.hndim))
        self.expander3=FermionHGen(spaceconfig=spaceconfig,evolutor=MaskedEvolutor(hndim=spaceconfig.hndim))

    def test_disc_symm(self,nsite=40):
        '''
        The parameters are adapted from PRB 54. 7598
        '''
        self.set_params(U=4.,t=1.,mu=0.,t2=0.,nsite=nsite)
        H_serial=op2collection(op=self.model_occ.hgen.get_opH())
        bmgen=get_bmgen(self.expander3.spaceconfig,'QM')
        dmrgegn=DMRGEngine(hchain=H_serial,hgen=self.expander3,tol=0,bmg=bmgen,symmetric=True,disc_symm='C')
        EG2=dmrgegn.run_finite(endpoint=(4,'->',nsite-2),maxN=[10,40,100,100,40],tol=0,block_params={'target_block':(0,0),'target_sector':{'C':-1}})[-1]
        print EG2*nsite
        pdb.set_trace()

    def test_nonint(self):
        #get the exact solution.
        h_exact=self.model_exact.hgen.H()
        E_excit=eigvalsh(h_exact)
        Emin_exact=sum(E_excit[E_excit<0])

        #the solution in occupation representation.
        h_occ=self.model_occ.hgen.H()
        Emin=eigsh(h_occ,which='SA',k=1)[0]
        print 'The Ground State Energy for hexagon(t = %s, t2 = %s) is %s, tolerence %s.'%(self.t,self.t2,Emin,Emin-Emin_exact)
        assert_almost_equal(Emin_exact,Emin)

        #the solution through updates
        H_serial=op2collection(op=self.model_occ.hgen.get_opH())
        H=get_H(H=H_serial,hgen=self.expander)
        H2,bm2=get_H_bm(H=H_serial,hgen=self.expander2,bstr='QM')
        Emin=eigsh(H,k=1,which='SA')[0]
        Emin2=eigsh(H2,k=1,which='SA')[0]
        print 'The Ground State Energy is %s, tolerence %s.'%(Emin,Emin-Emin2)
        assert_almost_equal(Emin_exact,Emin)
        assert_almost_equal(Emin_exact,Emin2)

        #the solution through dmrg.
        bmgen=get_bmgen(self.expander3.spaceconfig,'QM')
        dmrgegn=DMRGEngine(hchain=H_serial,hgen=self.expander3,tol=0,bmg=bmgen,symmetric=True)
        EG2=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=[10,20,30,40,40],tol=0)[-1]
        assert_almost_equal(Emin_exact,EG2*H_serial.nsite,decimal=4)

    def test_all(self):
        self.test_disc_symm(20)
        self.test_nonint()

TestFH().test_all()
