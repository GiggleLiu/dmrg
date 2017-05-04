from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh
import pdb,time,copy,sys
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,SuperSpaceConfig,SpaceConfig,RHGenerator,op_simple_hopping,op_simple_onsite,op_U
from tba.lattice import Chain
from pymps import WL2MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,\
        product_state,random_product_state,WL2OPC,check_validity_op,insert_Zs,op2collection
from dmrg import DMRGEngine
from blockmatrix import SimpleBMG

from vmps import VMPSEngine

class ChainN(object):
    '''This is a tight-binding model for a chain.'''
    def __init__(self,t,t2=0,U=0,mu=0.,occ=True,nsite=6):
        self.t,self.t2,self.U,self.mu=t,t2,U,mu
        self.occ=occ
        self.nsite=nsite

        #occupation representation will use <SuperSpaceConfig>, otherwise <SpaceConfig>.
        if self.occ:
            spaceconfig=SuperSpaceConfig([nsite,2,1])
        else:
            spaceconfig=SpaceConfig([2,nsite,1],kspace=False)
            if abs(U)>0: warnings.warn('U is ignored in non-occupation representation.')
        hgen=RHGenerator(spaceconfig=spaceconfig)

        #define the operator of the system
        hgen.register_params({
            '-t1':-self.t,
            '-t2':-self.t2,
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
        hgen.register_operator(op_t1,param='-t1')
        op_t2=op_simple_hopping(label='hop2',spaceconfig=spaceconfig,bonds=b2s)
        hgen.register_operator(op_t2,param='-t2')
        op_n=op_simple_onsite(label='n',spaceconfig=spaceconfig)
        hgen.register_operator(op_n,param='-mu')

        #add the hubbard interaction term if it is in the occupation number representation.
        if self.occ:
            op_ninj=op_U(label='ninj',spaceconfig=spaceconfig)
            hgen.register_operator(op_ninj,param='U')

        self.hgen=hgen
        spaceconfig1=SuperSpaceConfig([1,2,1])
        self.hchain=op2collection(op=self.hgen.get_opH())
        insert_Zs(self.hchain,spaceconfig=spaceconfig1)

def test_nonint():
    '''
    Run vMPS for Heisenberg model.
    '''
    nsite=10
    model=ChainN(t=1.,U=0.,nsite=nsite)
    spaceconfig1=SuperSpaceConfig([1,2,1])
    #exact result
    h0=zeros([nsite,nsite])
    fill_diagonal(h0[1:,:-1],1.); h0=h0+h0.T
    e_=2*eigvalsh(h0)
    e_exact=sum(e_[e_<0])
    pdb.set_trace()

    #run vmps
    #generate a random mps as initial vector
    bmg=SimpleBMG(spaceconfig=spaceconfig1,qstring='QM')
    #k0=product_state(config=random.randint(0,2,nsite),hndim=2)
    k0=product_state(config=repeat([1,2],nsite/2),hndim=spaceconfig1.hndim,bmg=bmg)
    mpo=model.hchain.toMPO(bmg=bmg,method='direct')
    #k0=product_state(config=repeat([0,2],nsite/2),hndim=model.spaceconfig.hndim,bmg=bmg)

    #setting up the engine
    vegn=VMPSEngine(H=mpo,k0=k0,eigen_solver='LC')
    #check the label setting is working properly
    assert_(all([ai.shape==(ai.labels[0].bm.N,ai.labels[1].bm.N,ai.labels[2].bm.N) for ai in vegn.ket.ML]))
    assert_(all([ai.shape==(ai.labels[0].bm.N,ai.labels[1].bm.N,ai.labels[2].bm.N,ai.labels[3].bm.N) for ai in vegn.H.OL]))
    #warm up
    vegn.warmup(10)
    vegn.eigen_solver='JD'
    e,v=vegn.run(maxN=[40,60,80,100,100,100,100],which='SA',nsite_update=2,endpoint=(5,'->',0))
    assert_almost_equal(e,e_exact,rtol=1e-4)

def test_int():
    '''
    Run vMPS for Heisenberg model.
    '''
    nsite=32
    U=10.0
    model=ChainN(t=1.,U=U,mu=U/2.,nsite=nsite)
    spaceconfig1=SuperSpaceConfig([1,2,1])
    #EG,mps=self.dmrgrun(model)

    #run vmps
    #generate a random mps as initial vector
    bmg=SimpleBMG(spaceconfig=spaceconfig1,qstring='QM')
    #k0=product_state(config=random.randint(0,2,nsite),hndim=2)
    k0=product_state(config=repeat([1,2],nsite/2),hndim=spaceconfig1.hndim,bmg=bmg)
    #mpo=model.hchain.toMPO(bmg=bmg,method='addition').compress(kernel='ldu')[0]
    #mpo.eliminate_zeros(1e-8)
    mpo=model.hchain.toMPO(bmg=bmg,method='direct')
    #mpo3=model.hchain.toMPO(bmg=bmg,method='direct').compress(kernel='ldu')[0]
    #mpo4=model.hchain.toMPO(bmg=bmg,method='direct').compress(kernel='dpl')[0]
    #k0=product_state(config=repeat([0,2],nsite/2),hndim=model.spaceconfig.hndim,bmg=bmg)

    #warm up the engine
    #vegn=VMPSEngine(H=mpo,k0=k0,eigen_solver='LC')
    #vegn.warmup(12)

    #generative run
    model2=ChainN(t=1,U=U,mu=U/2.,nsite=2)
    k02=product_state(config=array([1,2]),hndim=spaceconfig1.hndim,bmg=bmg)
    mpo2=model2.hchain.toMPO(bmg=bmg,method='direct')
    vegn=VMPSEngine(H=mpo2,k0=k02,eigen_solver='LC')
    vegn.generative_run(HP=mpo.OL[1:3],ngen=nsite/2-1,niter_inner=1,maxN=50,trunc_mps=False,which='SA')

    vegn.run(6,maxN=[40,50,50,50,50,50,50])
    #vegn.run(maxN=80,which='SA',nsite_update=1,endpoint=(3,'->',0))
    pdb.set_trace()

def test_inff(mode='save'):
    '''
    Run infinite vMPS for Heisenberg model.
    '''
    t=1.
    U=100.
    mu=U/2.
    #generate a random mps as initial vector
    spaceconfig1=SuperSpaceConfig([1,2,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig1,qstring='QM')
    k0=product_state(config=array([1,2]),hndim=spaceconfig1.hndim,bmg=bmg)

    model2=ChainN(t=t,U=U,nsite=2,mu=mu)
    mpo2=model2.hchain.toMPO(bmg=bmg,method='direct')

    model=ChainN(t=t,U=U,nsite=4,mu=mu)
    mpo4=model.hchain.toMPO(bmg=bmg,method='direct')

    mpo=model.hchain.toMPO(bmg=bmg,method='direct')
    #setting up the engine
    vegn=VMPSEngine(H=mpo2,k0=k0,eigen_solver='LC')
    vegn.run(1)
    #H2=mpo2.H
    #print eigvalsh(H2)[0]
    #print model2.hchain
    #print model2.hchain.H()-mpo2.H
    #pdb.set_trace()
    filetoken='data/con_dump_U%s_t%s_mu%s'%(U,t,mu)
    if mode=='save':
        vegn.generative_run(HP=mpo4.OL[1:3],ngen=100,niter_inner=1,maxN=100,trunc_mps=True,which='SA')
        vegn.con.dump_data(filetoken)
    else:
        vegn.con.load_data(filetoken)
    pdb.set_trace()


#test_nonint()
#test_int()
test_inff()
