'''
Variational Matrix Product State.
'''

from numpy import *
from scipy.linalg import svd,eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix,coo_matrix
from matplotlib.pyplot import *
import time,pdb

from rglib.mps import NORMAL_ORDER,Contraction,contract,Tensor,autoset_bms,svdbd,check_validity
from blockmatrix import trunc_bm
from pydavidson import JDh
from tba.hgen import ind2c
from flib.flib import fget_subblock

__all__=['VMPSEngine']

ZERO_REF=1e-12
DEBUG=True

def _evolve(mpo,ket,bra=None,nstep=None,fromleft=True):
    nsite=ket.nsite
    if nstep is None:nstep=nsite
    if bra is None:
        bra=ket.tobra(labels=[mpo.labels[0],ket.labels[1]+'\''])
    if fromleft:
        FL=Tensor(ones([1,1,1]),labels=[bra.get(0,attach_S='B').labels[0],mpo.get(0).labels[0],ket.get(0,attach_S='B').labels[0]])
    else:
        FL=Tensor(ones([1,1,1]),labels=[bra.get(nsite-1,attach_S='B').labels[2],mpo.get(nsite-1).labels[3],ket.get(nsite-1,attach_S='B').labels[2]])

    for i in xrange(nstep):
        ib=i if fromleft else nsite-i-1
        cbra=bra.get(ib,attach_S='B')
        cket=ket.get(ib,attach_S='B')
        ch=mpo.get(ib)
        FL=cbra*FL*ch*cket
    return FL

class VMPSEngine(object):
    '''
    Variational MPS Engine.

    Attributes:
        :H: <MPO>, the hamiltonian operator.
        :ket: <MPS>, the eigen state.
        :labels: list, they are [physical-bra, physical-ket, bond-bra, bond-mpo, bond-ket]
        :eigen_solver: str, eigenvalue solver.
            *'JD', Jacobi-Davidson method.
            *'LC', Lanczos, method.
    '''
    def __init__(self,H,k0,labels=['s','m','a','b','c'],eigen_solver='JD',bmg=None):
        self.H=H
        self.eigen_solver=eigen_solver
        self.bmg=bmg
        #set up initial ket
        self.ket=k0
        self.ket<<self.ket.l-1  #right normalize the ket to the first bond, where we start our update
        nsite=self.ket.nsite

        #unify labels
        self.labels=labels
        self.ket.chlabel([labels[1],labels[4]])
        self.H.chlabel([labels[0],labels[1],labels[3]])

        #setup block markers.
        if bmg is not None:
            autoset_bms(self.ket,bmg)
            autoset_bms(self.H,bmg)

        #initial contractions, fill the RPART(because we will start our update from left end)
        bra=self.ket.tobra(labels=[labels[0],labels[2]],sharedata=True)  #do not deepcopy the data

        #the contraction results for left and right parts, the size as the key.
        FL=Tensor(ones([1,1,1]),labels=[bra.get(0).labels[0],self.H.get(0).labels[0],self.ket.get(0).labels[0]])
        FR=Tensor(ones([1,1,1]),labels=[bra.get(nsite-1).labels[-1],self.H.get(nsite-1).labels[-1],self.ket.get(nsite-1).labels[-1]])
        self.LPART={0:FL}
        self.RPART={0:FR}
        for i in xrange(nsite-2):
            cbra=bra.get(nsite-i-1,attach_S='B')
            cket=self.ket.get(nsite-i-1,attach_S='B')
            ch=self.H.get(nsite-i-1)
            FR=cbra*FR*ch*cket
            self.RPART[i+1]=FR

    def _eigsh(self,H,v0,projector=None,tol=1e-10,sigma=None,lc_search_space=1,k=1,iprint=0):
        '''
        solve eigenvalue problem.
        '''
        maxiter=5000
        N=H.shape[0]
        if iprint==10 and projector is not None and check_commute:
            assert(is_commute(H,projector))
        if self.eigen_solver=='LC':
            k=max(lc_search_space,k)
            if H.shape[0]<100:
                e,v=eigh(H.toarray())
                e,v=e[:k],v[:,:k]
            else:
                try:
                    e,v=eigsh(H,k=k,which='SA',maxiter=maxiter,tol=tol,v0=v0)
                except:
                    e,v=eigsh(H,k=k+1,which='SA',maxiter=maxiter,tol=tol,v0=v0)
            order=argsort(e)
            e,v=e[order],v[:,order]
        else:
            iprint=0
            maxiter=500
            if projector is not None:
                e,v=JDh(H,v0=v0,k=k,projector=projector,tol=tol,maxiter=maxiter,sigma=sigma,which='SA',iprint=iprint)
            else:
                if sigma is None:
                    e,v=JDh(H,v0=v0,k=max(lc_search_space,k),projector=projector,tol=tol,maxiter=maxiter,which='SA',iprint=iprint)
                else:
                    e,v=JDh(H,v0=v0,k=k,projector=projector,tol=tol,sigma=sigma,which='SL',\
                            iprint=iprint,converge_bound=1e-10,maxiter=maxiter)

        nstate=len(e)
        if nstate==0:
            raise Exception('No Converged Pair!!')
        elif nstate==k or k>1:
            return e,v

        #filter out states meeting projector.
        if projector is not None and lc_search_space!=1:
            overlaps=array([abs(projector.dot(v[:,i]).conj().dot(v[:,i])) for i in xrange(nstate)])
            mask0=overlaps>0.1
            if not any(mask0):
                raise Exception('Can not find any states meeting specific parity!')
            mask=overlaps>0.9
            if sum(mask)==0:
                #check for degeneracy.
                istate=where(mask0)[0][0]
                warnings.warn('Wrong result or degeneracy accur!')
            else:
                istate=where(mask)[0][0]
            v=projector.dot(v[:,istate:istate+1])
            v=v/norm(v)
            return e[istate:istate+1],v
        else:
            #get the state with maximum overlap.
            v0H=v0.conj()/norm(v0)
            overlaps=array([abs(v0H.dot(v[:,i])) for i in xrange(nstate)])
            istate=argmax(overlaps)
            if overlaps[istate]<0.7:
                warnings.warn('Do not find any states same correspond to the one from last iteration!%s'%overlaps)
        e,v=e[istate:istate+1],v[:,istate:istate+1]
        return e,v

    def run(self,endpoint=(7,'->',1),maxN=50,tol=0,on_the_fly=True):
        '''
        Run this application.

        Parameters:
            :maxiter: int, the maximum sweep iteration, one iteration is contains one left sweep, and one right sweep through the chain.
            :maxN: list/int, the maximum kept dimension.
            :tol: float, the tolerence.
            :on_the_fly: bool, do not calculate the whole Hamiltonian if True.

        Return:
            (Emin, <MPS>)
        '''
        #check data
        maxiter=endpoint[0]+1
        nsite=self.ket.nsite
        hndim=self.ket.hndim
        if isinstance(maxN,int): maxN=[maxN]*maxiter
        if endpoint[1]=='->':
            assert(endpoint[2]>=1 and endpoint[2]<nsite)
        elif endpoint[1]=='<-':
            assert(endpoint[2]>=2 and endpoint[2]<nsite-1)
        else:
            raise ValueError()

        elist=[]
        for iiter in xrange(maxiter):
            print '########### STARTING NEW ITERATION %s ################'%iiter
            for direction,iterator in zip(['->','<-'],[xrange(1,nsite),xrange(nsite-2,1,-1)]):
                for l in iterator:   #l is the center(of two site) bond.
                    print 'Running iter = %s, direction = %s, l = %s'%(iiter,direction,l)
                    print 'A'*(l-1)+'..'+'B'*(nsite-l-1)
                    t0=time.time()

                    #construct the Tensor for Hamilonian
                    self.ket>>l-self.ket.l
                    FL=self.LPART[l-1]
                    FR=self.RPART[nsite-l-1]
                    O1,O2=self.H.get(l-1),self.H.get(l)
                    K10,K20=self.ket.get(l-1,attach_S='B'),self.ket.get(l,attach_S='B')
                    t00=time.time()
                    #get Tc,indices
                    if on_the_fly:
                        #get bmd = c(l-1)-m(l-1)-m(l)-c(l+1)
                        bms=[lb.bm for lb in K10.labels[:2]+K20.labels[-2:]]
                        bmd,pmd=self.bmg.join_bms(bms,signs=[1,1,1,-1])
                        #get the indices taken
                        sls=bmd.get_slice(zeros(len(self.bmg.qstring),dtype='int32'),uselabel=True)
                        indices=pmd[sls]
                        #turn the indices into subindices.
                        NN=array([bm.N for bm in bms])
                        cinds=ind2c(indices,NN)
                        #get the sub-block.
                        Tc=fget_subblock(FL,O1,O2,FR,cinds)
                    else:
                        T=(FL*O1*O2*FR).chorder([0,2,4,6,1,3,5,7])
                        dim=prod(T.shape[:4])
                        T=T.merge_axes(slice(0,4),bmg=self.bmg,signs=[1,1,1,-1],return_pm=False,compact_form=False)
                        T=T.merge_axes(slice(1,5),bmg=self.bmg,signs=[1,1,1,-1],return_pm=False,compact_form=False)
                        #get the specific block for T, the `charge` neutral block
                        #first, get the slice.
                        bmd,pmd=T.labels[1].bm.compact_form()
                        sls=bmd.get_slice(zeros(len(self.bmg.qstring),dtype='int32'),uselabel=True)
                        indices=pmd[sls]
                        #second, get the hamiltonian.
                        Tc=T[indices][:,indices]
                        Tc[abs(Tc)<ZERO_REF]=0
                    t1=time.time()

                    #third, get the initial vector
                    K1K20=(K10*K20).merge_axes(slice(0,2),bmg=self.bmg,signs=[1,1],compact_form=False).merge_axes(slice(1,3),bmg=self.bmg,signs=[-1,1],compact_form=False)
                    v0=asarray((K1K20).ravel())
                    v0c=v0[indices]
                    if DEBUG and not on_the_fly:
                        Et=_evolve(self.H,self.ket)
                        E0=v0.dot(T.dot(v0))
                        print 'Checking for energy expectation! E(true)=%s, E(now)=%s'%(Et,E0)
                        assert(abs(Et-E0)<1e-8)
                    Emin,K1K2c=self._eigsh(csr_matrix(Tc),v0=v0c,projector=None,tol=1e-10,sigma=None,lc_search_space=1,k=1)
                    K1K2=zeros(bmd.N,dtype='complex128')
                    K1K2[indices]=K1K2c
                    t2=time.time()

                    #update our ket
                    K1K2=Tensor(K1K2.reshape([FL.shape[0]*hndim,FR.shape[0]*hndim]),labels=K1K20.labels)
                    #make it block diagonal
                    K1K2_block,pms=K1K2.b_reorder(return_pm=True)
                    #perform svd and roll back to original non-block structure
                    K1,S,K2=svdbd(K1K2_block,cbond_str='%s_%s'%(self.labels[-1],l))
                    K1,K2=K1[argsort(pms[0])],K2[:,argsort(pms[1])]
                    #do the truncation
                    if len(S)>maxN[iiter]:
                        Smin=sort(S)[-maxN[iiter]]
                        kpmask=S>=Smin
                        K1,S,K2=K1[:,kpmask],S[kpmask],K2[kpmask]
                        bm_new=trunc_bm(K2.labels[0].bm,kpmask=kpmask)
                        K2.labels[0].bm=K1.labels[1].bm=bm_new
                    K1[abs(K1)<ZERO_REF]=0
                    K2[abs(K2)<ZERO_REF]=0
                    bdim=len(S)
                    self.ket.AL[-1],self.ket.S,self.ket.BL[0]=Tensor(K1.reshape([K10.shape[0],hndim,bdim]),labels=K10.labels[:2]+K1.labels[-1:]),\
                            S,Tensor(K2.reshape([bdim,hndim,K20.shape[2]]),labels=K2.labels[:1]+K20.labels[1:])

                    #update our contractions.
                    #1. the left part
                    cket=self.ket.get(l-1,attach_S='B')
                    cbra=cket.conj()
                    cbra.labels=[cket.labels[0].chstr('%s_%s'%(self.labels[2],l-1)),\
                            cket.labels[1].chstr('%s_%s'%(self.labels[0],l-1)),\
                            cket.labels[2].chstr('%s_%s'%(self.labels[2],l))]
                    self.LPART[l]=cbra*self.LPART[l-1]*self.H.get(l-1)*cket

                    #2. the right part
                    cket=self.ket.get(l,attach_S='A')
                    cbra=cket.conj()
                    cbra.labels=[cket.labels[0].chstr('%s_%s'%(self.labels[2],l)),\
                            cket.labels[1].chstr('%s_%s'%(self.labels[0],l)),\
                            cket.labels[2].chstr('%s_%s'%(self.labels[2],l+1))]
                    self.RPART[nsite-l]=cbra*self.RPART[nsite-l-1]*self.H.get(l)*cket
                    t3=time.time()

                    #schedular check
                    elist.append(Emin)
                    diff=Inf if len(elist)<=1 else elist[-1]-elist[-2]
                    print 'Get Emin = %.12f, tol = %s, Elapse -> %s, %s states kept.'%(Emin,diff,t3-t0,len(S))
                    print 'Time: get Tc(%s), eigen(%s), svd(%s)'%(t1-t0,t2-t1,t3-t2)
                    if iiter==endpoint[0] and direction==endpoint[1] and l==endpoint[2]:
                        print 'RUN COMPLETE!'
                        return Emin,self.ket

            #schedular check for each iteration
            if iiter==0:
                Emin_last=Inf
            diff=Emin_last-Emin
            print 'ITERATION SUMMARY: Emin/site = %s, tol = %s'%(Emin,diff)
