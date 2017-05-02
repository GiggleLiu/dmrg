from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh,svd,norm
import pdb,time,copy,sys
sys.path.insert(0,'../')

from tba.hgen import SpinSpaceConfig,SuperSpaceConfig,SpaceConfig,RHGenerator,op_simple_hopping,op_simple_onsite,op_U,quickload
from tba.lattice import Chain
from pymps import WL2MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,\
        product_state,random_product_state,WL2OPC,check_validity_op,insert_Zs,op2collection,Tensor
from dmrg import DMRGEngine
from blockmatrix import SimpleBMG

from vmps import VMPSEngine

random.seed(2)

def test_inff(mode='save'):
    '''
    Run infinite vMPS for Heisenberg model.
    '''
    t=1.
    U=2.
    mu=U/2.
    #generate a random mps as initial vector
    spaceconfig1=SuperSpaceConfig([1,2,1])
    bmg=SimpleBMG(spaceconfig=spaceconfig1,qstring='QM')
    k0=product_state(config=array([1,2]),hndim=spaceconfig1.hndim,bmg=bmg)

    model2=ChainN(t=t,U=U,nsite=2,mu=mu)
    mpo2=model2.hchain.toMPO(bmg=bmg,method='direct')

    model=ChainN(t=t,U=U,nsite=4)
    mpo4=model.hchain.toMPO(bmg=bmg,method='direct')

    mpo=model.hchain.toMPO(bmg=bmg,method='direct')
    #setting up the engine
    vegn=VMPSEngine(H=mpo2,k0=k0,eigen_solver='LC')
    if mode=='save':
        vegn.generative_run(HP=mpo4.OL[1:3],ngen=1000,niter_inner=1,maxN=200,trunc_mps=True,which='SA')
        filetoken='test_contractor_dump'
        vegn.con.dump_data(filetoken)
    else:
        vegn.con.load_data(filetoken)
    pdb.set_trace()


def generate_env(ts):
    return Tensor(identity(ts.shape[0]),[ts.labels[0]+'-conj',ts.labels[0]]),\
            Tensor(identity(ts.shape[-1]),[ts.labels[-1]+'-conj',ts.labels[-1]])

def entropy(rho):
    ZERO_REF=1e-12
    U,S,V=svd(rho,full_matrices=False)
    print 'norms = %s.'%sum(S)
    S=S[S>ZERO_REF]
    S=S/sum(S)
    return -sum(S*log(S))

def get_mutual_information(ts,env):
    '''
    mutual information of s1, s2.

    Parameters:
        :ts: <Tensor>, with 4 labels [al,s1,s2,ar].
        :env: tuple of 2 <Tensor>, with 2 labels [al,al']
    '''
    envl,envr=env
    tsbra=ts.make_copy(labels=[envl.labels[0],ts.labels[1]+'_conj',\
            ts.labels[2]+'_conj',envr.labels[0]]).conj()
    TR=ts*envr
    rho12=envl*tsbra*TR
    rho12=rho12.reshape([rho12.shape[0]*rho12.shape[1],-1])
    tsbra.chlabel(ts.labels[2],axis=2)
    rho1=envl*tsbra*ts*envr
    tsbra.chlabel(ts.labels[2]+'_conj',axis=2)
    tsbra.chlabel(ts.labels[1],axis=1)
    rho2=envl*tsbra*ts*envr

    #calculate mutual information
    mutual_info=entropy(rho1)+entropy(rho2)-entropy(rho12)
    return mutual_info

def US_mutual_info(US,sndim1=2,sndim2=2):
    nr=US.shape[-1]
    t=Tensor(US.reshape([-1,sndim1,sndim2,nr]),labels=['al','m1','m2','ar'])
    envl,envr=generate_env(t)
    mutual_info=get_mutual_information(ts=t,env=[envl,envr])
    return mutual_info

def quick_mutual_info(U,S,sndim1=2,sndim2=2):
    '''
    mutual information of s1, s2.

    Parameters:
        :U,S: U,S in MPS.
    '''
    US=U.mul_axis(S,axis=-1)
    nr=US.shape[-1]
    ts=Tensor(US.reshape([-1,sndim1,sndim2,nr]),labels=['al','m1','m2','ar'])
    tsbra=ts.make_copy(labels=[ts.labels[0],'s1',
            's2',ts.labels[3]]).conj()
    rho12=tsbra*ts
    rho1=trace(rho12,axis1=1,axis2=3)
    rho2=trace(rho12,axis1=0,axis2=2)
    rho12=rho12.reshape([rho12.shape[0]*rho12.shape[1],-1])

    #calculate mutual information
    mutual_info=entropy(rho1)+entropy(rho2)-entropy(rho12)
    return mutual_info


def test_mutual():
    #generate test case
    nl=nrr=50
    K=random.random([nl*4,nrr*4])+1j*random.random([nl*4,nrr*4])-0.5-0.5j
    U,S,V=svd(K,full_matrices=False)
    S=S/norm(S)

    mutual_info=US_mutual_info(U*S)
    print 'mutual_info = %s'%mutual_info
    pdb.set_trace()

def hubbard_mutual():
    U=2.0
    t=1.0
    mu=U/2.
    filetoken='con_dump_U%s_t%s_mu%s'%(U,t,mu)
    mps=quickload(filetoken+'.mps.dat')
    mutual_info=quick_mutual_info(mps.get(0),mps.S)
    #mutual_info=US_mutual_info(mps.get(0).mul_axis(mps.S,axis=-1))
    print 'mutual_info = %s'%mutual_info
    pdb.set_trace()

#test_mutual()
hubbard_mutual()
