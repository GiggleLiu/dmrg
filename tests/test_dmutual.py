from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh,svd,norm
from scipy.optimize import minimize
import pdb,time,copy,sys
sys.path.insert(0,'../')

from test_unitary import U4SPH,U4,U4U2
from pymps import WL2MPO,OpUnitI,opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy,MPS,\
        product_state,random_product_state,WL2OPC,check_validity_op,insert_Zs,op2collection,Tensor
from tba.hgen import SpinSpaceConfig,SuperSpaceConfig,SpaceConfig,RHGenerator,op_simple_hopping,op_simple_onsite,op_U,quickload

#random.seed(4)
def generate_env(ts):
    return Tensor(identity(ts.shape[0]),[ts.labels[0]+'-conj',ts.labels[0]]),\
            Tensor(identity(ts.shape[-1]),[ts.labels[-1]+'-conj',ts.labels[-1]])

def entropy(rho):
    ZERO_REF=1e-12
    U,S,V=svd(rho,full_matrices=False)
    #print 'norms = %s.'%sum(S)
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

def quick_mutual_info(US,sndim1=2,sndim2=2):
    '''
    mutual information of s1, s2.

    Parameters:
        :US: U*S in MPS.
    '''
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

#apply unitary matrix to tensor, and get mutual information
def Uni_mutual_info(US,x):
    umat=Tensor(UGEN(x),labels=[US.labels[1],US.labels[1]+'_'])  #18 parameters
    US=(umat*US).chorder([1,0,2])

    mutual_info=quick_mutual_info(US)
    #print 'mutual_info = %s'%mutual_info
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

def hubbard_mutual(U=10.):
    t=1.0
    mu=U/2.
    filetoken='data/con_dump_U%s_t%s_mu%s'%(U,t,mu)
    mps=quickload(filetoken+'.mps.dat')

    U0S=mps.get(0).mul_axis(mps.S,axis=-1)
    x0=random.random(ULEN)
    #x0=zeros(ULEN)
    opt=minimize(lambda x:Uni_mutual_info(U0S,x),x0=x0,method='COBYLA',tol=1e-20)
    print 'initial- x=%s, info=%s\nfinal- x=%s, cost=%s.'%(x0,Uni_mutual_info(U0S,x0),opt.x,opt.fun)
    #savetxt('data/x%s_U%s.dat'%(ULEN,U),opt.x)
    return opt.x,opt.fun

def analyse_U(U=10.):
    x=loadtxt('data/x%s_U%s.dat'%(ULEN,U))
    Umat=UGEN(x[:6],x[6:])
    print abs(Umat)
    pdb.set_trace()

def eigen_analysis():
    t=1.0
    UL=[0.,0.5,1.,2.,10.,100.]
    eigen_vals=[]
    sndim1=sndim2=2
    for U in UL:
        mu=U/2.
        filetoken='data/con_dump_U%s_t%s_mu%s'%(U,t,mu)
        mps=quickload(filetoken+'.mps.dat')

        US=mps.get(0).mul_axis(mps.S,axis=-1)
        nr=US.shape[-1]
        ts=Tensor(US.reshape([-1,sndim1,sndim2,nr]),labels=['al','m1','m2','ar'])
        tsbra=ts.make_copy(labels=[ts.labels[0],'s1',
                's2',ts.labels[3]]).conj()
        rho12=tsbra*ts
        rho1=trace(rho12,axis1=1,axis2=3)
        rho2=trace(rho12,axis1=0,axis2=2)
        rho12=rho12.reshape([rho12.shape[0]*rho12.shape[1],-1])
        eigen_vals.append(eigvalsh(rho12))
    ion()
    plot(UL,eigen_vals)
    pdb.set_trace()

def updown_analysis():
    t=1.0
    UL=[0.,0.5,1.,2.,10.,100.]
    mutual_info=[]
    sndim1=sndim2=2
    for U in UL:
        mu=U/2.
        filetoken='data/con_dump_U%s_t%s_mu%s'%(U,t,mu)
        mps=quickload(filetoken+'.mps.dat')

        US=mps.get(0).mul_axis(mps.S,axis=-1)
        mi=quick_mutual_info(US)
        mutual_info.append(mi)
    ion()
    fig=figure(figsize=(5,4))
    plot(UL,mutual_info)
    xscale('log')
    xlabel(r'$U$',fontsize=16)
    ylabel(r'$I(\rho_\uparrow,\rho_\downarrow)$',fontsize=16)
    fig.tight_layout()
    pdb.set_trace()

def uni2_analysis():
    t=1.0
    UL=[0.,0.5,1.,2.,10.,100.]
    mutual_info=[]
    sndim1=sndim2=2
    for U in UL:
        x_final,mi=hubbard_mutual(U)
        mutual_info.append(mi)
    ion()
    fig=figure(figsize=(5,4))
    plot(UL,mutual_info)
    xscale('log')
    xlabel(r'$U$',fontsize=16)
    ylabel(r'$I(\rho_{\gamma_1},\rho_{\gamma_2})$',fontsize=16)
    fig.tight_layout()
    pdb.set_trace()


UGEN=U4U2
ULEN=3
#test_mutual()
#hubbard_mutual(10.)
#analyse_U(2.)
#eigen_analysis()
#updown_analysis()
uni2_analysis()
