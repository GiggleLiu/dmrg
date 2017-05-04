from numpy import *
from matplotlib.pyplot import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh,svd,norm,det
import pdb,time,copy,sys
sys.path.insert(0,'../')

def U4SPH(params):
    ths,phs=split(params,[6])
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

def U4(params):
    ths,phs=split(params,[6])
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

    d4=exp(1j*phs[:4])
    d13=append([1],exp(1j*phs[4:7]))
    d22=append([1]*2,exp(1j*phs[7:9]))
    d31=append([1]*3,exp(1j*phs[9:10]))

    return (d4[:,newaxis]*O4*d13).dot(O13*d22).dot(O22*d31)

def U4U2(params):
    theta,phi1,phi2=params
    a,b=cos(theta)*exp(1j*phi1),sin(theta)*exp(1j*phi2)
    ac,bc=a.conj(),b.conj()
    U2=array([[a,b],[-bc,ac]])
    res=zeros([4,4],dtype='complex128')
    res[0,0]=res[3,3]=1
    res[1:3,1:3]=U2
    return res

def test_U4():
    dt1=dt2=0
    ntest=20
    for i in xrange(ntest):
        t0=time.time()
        U44=U4(random.random(16)*2*pi)
        t1=time.time()
        U44SPH=U4SPH(random.random(18)*2*pi)
        t2=time.time()
        dt1+=t1-t0
        dt2+=t2-t1
        U44U2=U4U2(random.random(3)*2*pi)
        res1=U44.dot(U44.T.conj())
        res2=U44SPH.dot(U44SPH.T.conj())
        res3=U44U2.dot(U44U2.T.conj())
        print 'Det = %s, %s(spherical)'%(det(U44),det(U44SPH))
        assert_allclose(res1,identity(4),atol=1e-8)
        assert_allclose(res2,identity(4),atol=1e-8)
        assert_allclose(res3,identity(4),atol=1e-8)
    print 'Elapse',dt1,dt2

if __name__=='__main__':
    random.seed(2)
    test_U4()
