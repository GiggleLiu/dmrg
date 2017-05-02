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

def U4(ths,phs):
    arr=zeros(12,dtype='complex128')
    arr[::2]=cos(ths)*exp(1j*phs[::2])
    arr[1::2]=sin(ths)*exp(1j*phs[1::2])
    a,b,c,d,e,f,g,h,j,k,l,m=arr
    ac,bc,cc,dc,ec,fc,gc,hc,jc,kc,lc,mc=arr.conj()
    m3a=-ac*d*hc*j+cc*gc*j*l-cc*kc*mc
    m3b=-gc*j*m-kc*lc
    m4a=-ac*d*hc*k+cc*gc*k*l+cc*jc*mc
    m4b=-gc*k*m+jc*lc
    return array([[a,b*c,b*d*e,b*d*f],
        [bc*g,-ac*c*g+dc*h*l,-ac*d*e*g-cc*e*h*l+fc*h*m,-ac*d*f*g-cc*f*h*l-ec*h*m],
        [bc*hc*j,-ac*c*hc*j-dc*gc*j*l+dc*kc*mc,m3a*e+m3b*fc,m3a*f-m3b*ec],
        [bc*hc*k,-ac*c*hc*k-dc*gc*k*l-dc*jc*mc,m4a*e+m4b*fc,m4a*f-m4b*ec]
        ])

def U4(ths,phs):
    c1,c2,c3,c4,c5,c6=cos(ths)
    s1,s2,s3,s4,s5,s6=sin(ths)
    O4=[[c1,-s1,0,0],
            [s1*c2,c1*c2,-s2,0],
            [s1*s2*c3,c1*s2*c3,c2*c3,-s3],
            [s1*s2*s3,c1*s2*s3,c2*s3,c3]]
    O13=[[1,0,0,0],
            [0,c4,-s4,0],
            [0,s4*c5,c4*c5,-s5],
            [0,s4*s5,c4*s5,c5],
            ]
    O22=zeros([4,4])
    O22[0,0]=O22[1,1]=1
    O22[2:,2:]=[[c6,-s6],[s6,c6]]

    d4=diag(exp(1j*phs[:4]))
    d13=diag(append([1],exp(1j*phs[4:7])))
    d22=diag(append([1]*2,exp(1j*phs[7:9])))
    d31=diag(append([1]*3,exp(1j*phs[9:])))

    return d4.dot(O4).dot(d13).dot(O13).dot(d22).dot(O22).dot(d31)

def test_U4():
    #U44=U4(random.random(6)*2*pi,random.random(12)*2*pi)
    U44=U4(random.random(6)*2*pi,random.random(10)*2*pi)
    res=U44.dot(U44.T.conj())
    assert_allclose(res,identity(4),atol=1e-8)

test_U4()
