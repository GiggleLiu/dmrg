'''
Test program for DMRG fermionic solver.
'''

from numpy import *
from matplotlib.pyplot import *
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import scipy.sparse as sps
import time,pdb,warnings,copy

from tba.hgen import RHGenerator,op_simple_hopping,op_U,SuperSpaceConfig,SpaceConfig,op_simple_onsite
from tba.lattice import Chain
from rglib.mps import op2collection
from rglib.hexpand import RGHGen,NullEvolutor,MaskedEvolutor,Evolutor
from lanczos import get_H,get_H_bm
from dmrg import *

swap_axis=True
if swap_axis:
    SpaceConfig.SPACE_TOKENS=['nambu','atom','spin','orbit']

def chorder(l):
    '''change the order of config.'''
    if swap_axis:
        l[-3],l[-2]=l[-2],l[-3]
    return l

class ChainN(object):
    '''This is a tight-binding model for a chain.'''
    def __init__(self,t,t2=0,U=0,mu=0.,occ=True,nsite=6):
        self.t,self.t2,self.U,self.mu=t,t2,U,mu
        self.occ=occ
        self.nsite=nsite

        #occupation representation will use <SuperSpaceConfig>, otherwise <SpaceConfig>.
        if self.occ:
            spaceconfig=SuperSpaceConfig(chorder([1,2,nsite,1]))
        else:
            spaceconfig=SpaceConfig(chorder([1,2,nsite,1]),kspace=False)
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
        self.model_exact=ChainN(t=t,U=0,t2=t2,mu=mu,occ=False,nsite=nsite)
        self.model_occ=ChainN(t=t,U=U,t2=t2,mu=mu,occ=True,nsite=nsite)
        scfg=self.model_occ.hgen.spaceconfig
        self.spaceconfig1=SuperSpaceConfig(chorder([scfg.nspin,1,scfg.norbit]))

    def test_disc_symm(self,nsite=40):
        '''
        The parameters are adapted from PRB 54. 7598
        '''
        self.set_params(U=2.,t=1.,mu=1.,t2=0.,nsite=nsite)
        spaceconfig=self.spaceconfig1
        H_serial=op2collection(op=self.model_occ.hgen.get_opH())
        expander3=RGHGen(spaceconfig=spaceconfig,H=H_serial,evolutor_type='masked',use_zstring=True)
        dmrgegn=DMRGEngine(hgen=expander3,tol=0,reflect=True)
        dmrgegn.use_U1_symmetry('QM',target_block=(0,0))
        for c in [-1,1]:
            dmrgegn.use_disc_symmetry(target_sector={'C':c},detect_scope=4)
            EG2,EV2=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=[20,40,50,70,70],tol=0)
            print 'Get gound state energy for C2 -> %s: %s.'%(c,EG2)
        #the result is -36.1372 for C=-1, and -36.3414 for C=1
        pdb.set_trace()

    def test_nonint(self):
        #get the exact solution.
        spaceconfig=self.spaceconfig1
        self.set_params(U=0.,t=1.,mu=0.2,t2=0.,nsite=6)
        h_exact=self.model_exact.hgen.H()
        E_excit=eigvalsh(h_exact)
        Emin_exact=sum(E_excit[E_excit<0])

        #the solution in occupation representation.
        h_occ=self.model_occ.hgen.H()
        Emin,Vmin1=eigsh(h_occ,which='SA',k=1)
        print 'The Ground State Energy for hexagon(t = %s, t2 = %s) is %s, tolerence %s.'%(self.t,self.t2,Emin,Emin-Emin_exact)
        assert_almost_equal(Emin_exact,Emin)

        #the solution through updates
        H_serial=op2collection(op=self.model_occ.hgen.get_opH())
        H_serial_Z=copy.copy(H_serial)
        H_serial_Z.insert_Zs(spaceconfig=spaceconfig)
        expander=RGHGen(spaceconfig=spaceconfig,H=H_serial_Z,evolutor_type='null',use_zstring=True)
        H=get_H(hgen=expander)
        expander2=RGHGen(spaceconfig=spaceconfig,H=H_serial_Z,evolutor_type='normal',use_zstring=True)
        H2,bm2=get_H_bm(hgen=expander2,bstr='QM')
        Emin=eigsh(H,k=1,which='SA')[0]
        Emin2=eigsh(H2,k=1,which='SA')[0]
        print 'The Ground State Energy is %s, tolerence %s.'%(Emin,Emin-Emin2)
        assert_almost_equal(Emin_exact,Emin)
        assert_almost_equal(Emin_exact,Emin2)

        #the solution through dmrg.
        expander3=RGHGen(spaceconfig=spaceconfig,H=H_serial,evolutor_type='masked',use_zstring=True)
        dmrgegn=DMRGEngine(hgen=expander3,tol=0,reflect=False)
        dmrgegn.use_U1_symmetry('QM',target_block=(0,0))
        EG2,Vmin2=dmrgegn.run_finite(endpoint=(5,'<-',0),maxN=[10,20,30,40,40],tol=0)
        Vmin2=fix_tail(Vmin2,expander3.spaceconfig,0)
        #check for states.
        assert_almost_equal(Emin_exact,EG2,decimal=4)
        Vmin1=Vmin1[:,0]
        assert_almost_equal(abs(Vmin2.state),abs(Vmin1),decimal=3)
        print (Vmin2.state/Vmin1)[abs(Vmin1)>1e-2]
        pdb.set_trace()

    def test_site_image(self):
        H_serial=op2collection(op=self.model_occ.hgen.get_opH())
        H_serial.insert_Zs(spaceconfig=self.spaceconfig1)
        H2=site_image(H_serial,care_sign=False,NL=0,NR=self.model_exact.nsite)
        print H_serial
        print H2

    def test_all(self):
        self.test_nonint()
        self.test_disc_symm(20)
        self.test_site_image()


TestFH().test_all()
