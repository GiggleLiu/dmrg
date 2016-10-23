'''
Variational Matrix Product State.
'''

from numpy import *
from scipy.linalg import svd,eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix,coo_matrix
from matplotlib.pyplot import *
import time,pdb

from rglib.mps import Contraction,contract,Tensor,svdbd,check_validity_mps,BLabel,BMPS,check_flow_mpx
from blockmatrix import trunc_bm
from pydavidson import JDh
from tba.hgen import ind2c
from flib.flib import fget_subblock2a,fget_subblock2b,fget_subblock1

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
    def __init__(self,H,k0,labels=['s','m','a','b','c'],eigen_solver='JD'):
        self.H=H
        self.eigen_solver=eigen_solver
        #set up initial ket
        self.ket=k0
        self.ket<<self.ket.l-1  #right normalize the ket to the first bond, where we start our update
        nsite=self.ket.nsite

        #unify labels
        self.labels=labels
        self.ket.chlabel([labels[1],labels[4]])
        self.H.chlabel([labels[0],labels[1],labels[3]])

        #initial contractions, fill the RPART(because we will start our update from left end)
        bra=self.ket.tobra(labels=[labels[0],labels[2]],sharedata=True)  #do not deepcopy the data

        #the contraction results for left and right parts, the size as the key.
        FL=Tensor(ones([1,1,1]),labels=[bra.get(0).labels[0],self.H.get(0).labels[0],self.ket.get(0).labels[0]])
        FR=Tensor(ones([1,1,1]),labels=[bra.get(nsite-1).labels[-1],self.H.get(nsite-1).labels[-1],self.ket.get(nsite-1).labels[-1]])
        self.LPART={0:FL}
        self.RPART={0:FR}
        for i in xrange(nsite-1):
            cbra=bra.get(nsite-i-1,attach_S='B')
            cket=self.ket.get(nsite-i-1,attach_S='B')
            ch=self.H.get(nsite-i-1)
            FR=cbra*FR*ch*cket
            self.RPART[i+1]=FR

    @property
    def energy(self):
        '''Get the energy from current status.'''
        return _evolve(self.H,self.ket,fromleft=False).item()

    def _eigsh(self,H,v0,projector=None,tol=1e-10,sigma=None,lc_search_space=1,k=1,iprint=0,which='SA'):
        '''
        solve eigenvalue problem.
        '''
        maxiter=5000
        N=H.shape[0]
        if iprint==10 and projector is not None:
            assert(is_commute(H,projector))
        if which=='SL':
            E,V=eigh(H)
            #get the eigenvector with maximum overlap
            overlap=reshape(v0,[1,-1]).dot(V).ravel()
            ind=argmax(overlap)
            print 'Overlap = %s'%overlap[ind]
            return E[ind],V[:,ind:ind+1]
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

    def run(self,endpoint=(7,'->',0),maxN=50,tol=0,which='SL',nsite_update=2):
        '''
        Run this application.

        Parameters:
            :maxiter: int, the maximum sweep iteration, one iteration is contains one left sweep, and one right sweep through the chain.
            :maxN: list/int, the maximum kept dimension.
            :tol: float, the tolerence.
            :which: str, string that specify the <MPS> desired, 'SL'(most similar to k0) or 'SA'(smallest)
            :nsite_update: int, the sites updated one time.

        Return:
            (Emin, <MPS>)
        '''
        #check data
        maxiter=endpoint[0]+1
        nsite=self.ket.nsite
        hndim=self.ket.hndim
        bmg=self.ket.bmg if hasattr(self.ket,'bmg') else None
        if isinstance(maxN,int): maxN=[maxN]*maxiter
        if endpoint[1]=='->':
            assert(endpoint[2]>=0 and endpoint[2]<nsite-nsite_update)
        elif endpoint[1]=='<-':
            assert(endpoint[2]>0 and endpoint[2]<=nsite-nsite_update)
        else:
            raise ValueError()

        elist=[]
        for iiter in xrange(maxiter):
            print '########### STARTING NEW ITERATION %s ################'%iiter
            for direction,iterator in zip(['->','<-'],[xrange(nsite-nsite_update),xrange(nsite-nsite_update,0,-1)]):
                for l in iterator:   #l is the index of first site.
                    print 'Running iter = %s, direction = %s, l = %s'%(iiter+1,direction,l)
                    print 'A'*(l)+'.'*nsite_update+'B'*(nsite-l-nsite_update)
                    t0=time.time()

                    #construct the Tensor for Hamilonian
                    self.ket>>l+nsite_update/2-self.ket.l
                    FL=self.LPART[l]
                    FR=self.RPART[nsite-l-nsite_update]
                    Os=[self.H.get(li) for li in xrange(l,l+nsite_update)]
                    K0s=[self.ket.get(li,attach_S='B') for li in xrange(l,l+nsite_update)]
                    t00=time.time()
                    #get Tc,indices
                    #get bmd = c(l-1)-m(l-1)-m(l)-c(l+1)
                    bms=[lb.bm for lb in K0s[0].labels[:1]+[K.labels[1] for K in K0s]+K0s[-1].labels[-1:]]
                    bmd,pmd=bmg.join_bms(bms,signs=[1]+[1]*nsite_update+[-1])
                    #get the indices taken
                    sls=bmd.get_slice(zeros(len(bmg.qstring),dtype='int32'),uselabel=True)
                    indices=pmd[sls]
                    #turn the indices into subindices.
                    NN=array([bm.N for bm in bms])
                    cinds=ind2c(indices,NN)
                    #get the sub-block.
                    if nsite_update==2:
                        Tc=fget_subblock2b(FL,Os[0],Os[1],FR,cinds)
                    elif nsite_update==1:
                        Tc=fget_subblock1(FL,Os[0],FR,cinds)
                    t1=time.time()

                    #third, get the initial vector
                    V0=reduce(lambda x,y:x*y,K0s)
                    v0=asarray(V0.ravel())
                    v0c=v0[indices]
                    if which=='SA':
                        Emin,Vc=self._eigsh(csr_matrix(Tc),v0=v0c,projector=None,tol=1e-10,sigma=None,lc_search_space=1,k=1)
                    elif which=='SL':
                        Emin,Vc=self._eigsh(Tc,v0=v0c,which='SL')
                    else:
                        raise ValueError()
                    V=zeros(bmd.N,dtype='complex128')
                    V[indices]=Vc
                    overlap=v0.dot(V)
                    t2=time.time()

                    #update our ket,
                    if nsite_update==2:
                        #if 2 site update, perform svd and truncate.
                        #first, find the block structure and make it block diagonal
                        bml=bmg.join_bms([V0.labels[0].bm,V0.labels[1].bm],signs=[1,1],compact_form=False)[0]
                        bmr=bmg.join_bms([V0.labels[2].bm,V0.labels[3].bm],signs=[-1,1],compact_form=False)[0]
                        K1K2=Tensor(V.reshape([FL.shape[0]*hndim,FR.shape[0]*hndim]),labels=[BLabel('KL',bml),BLabel('KR',bmr)])
                        K1K2_block,pms=K1K2.b_reorder(return_pm=True)
                        #perform svd and roll back to original non-block structure
                        K1,S,K2=svdbd(K1K2_block,cbond_str='%s_%s'%(self.labels[-1],l+1))
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
                        #set datas
                        bdim=len(S)
                        self.ket.AL[-1]=Tensor(K1.reshape([K0s[0].shape[0],hndim,bdim]),labels=K0s[0].labels[:2]+K1.labels[-1:])
                        self.ket.S=S
                        self.ket.BL[0]=Tensor(K2.reshape([bdim,hndim,K0s[1].shape[2]]),labels=K2.labels[:1]+K0s[1].labels[1:])
                    elif nsite_update==1:
                        if direction=='->':
                            self.ket.BL[0]=Tensor(V.reshape(V0.shape),labels=V0.labels)
                            self.ket.S=ones(V0.shape[0])  #S has been taken into consideration, so, don't use it anymore.
                            self.ket>>1
                        else:
                            self.ket.AL.append(Tensor(V.reshape(V0.shape),labels=V0.labels)); self.ket.BL.pop(0)
                            self.ket.S=ones(V0.shape[-1])  #S has been taken into consideration, so, don't use it anymore.
                            self.ket<<1
                        bdim=len(self.ket.S)

                    #update our contractions.
                    #1. the left part
                    l_left=l if nsite_update==2 else self.ket.l-1
                    cket=self.ket.get(l_left)
                    cbra=cket.conj()
                    cbra.labels=[cket.labels[0].chstr('%s_%s'%(self.labels[2],l_left)),\
                            cket.labels[1].chstr('%s_%s'%(self.labels[0],l_left)),\
                            cket.labels[2].chstr('%s_%s'%(self.labels[2],l_left+1))]
                    t3=time.time()
                    self.LPART[l_left+1]=cbra*self.LPART[l_left]*self.H.get(l_left)*cket
                    print 'update L=%s, shape %s'%(l_left+1,self.LPART[l_left+1].shape)

                    #2. the right part
                    l_right=l+1 if nsite_update==2 else self.ket.l
                    cket=self.ket.get(l_right)
                    cbra=cket.conj()
                    cbra.labels=[cket.labels[0].chstr('%s_%s'%(self.labels[2],l_right)),\
                            cket.labels[1].chstr('%s_%s'%(self.labels[0],l_right)),\
                            cket.labels[2].chstr('%s_%s'%(self.labels[2],l_right+1))]
                    self.RPART[nsite-l_right]=cbra*self.RPART[nsite-l_right-1]*self.H.get(l_right)*cket

                    #schedular check
                    elist.append(Emin)
                    diff=Inf if len(elist)<=1 else elist[-1]-elist[-2]
                    print 'Get Emin = %.12f, tol = %s, Elapse -> %s, %s states kept, overlap %s.'%(Emin,diff,t3-t0,bdim,overlap)
                    print 'Time: get Tc(%s), eigen(%s), svd(%s)'%(t1-t0,t2-t1,t3-t2)
                    if bdim>maxN[-1] and isinstance(bdim,int): pdb.set_trace()
                    if iiter==endpoint[0] and direction==endpoint[1] and l==endpoint[2]:
                        print 'RUN COMPLETE!'
                        return Emin,self.ket

            #schedular check for each iteration
            if iiter==0:
                Emin_last=Inf
            diff=Emin_last-Emin
            Emin_last=Emin
            print 'ITERATION SUMMARY: Emin/site = %s, tol = %s'%(Emin,diff)
