'''
Variational Matrix Product State.
'''

from numpy import *
from scipy.linalg import svd,eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix,coo_matrix,csc_matrix
from scipy.sparse import kron as skron
from matplotlib.pyplot import *
import time,pdb

from pymps import contract,Tensor,check_validity_mps,BLabel,check_flow_mpx,get_sweeper
from contractor import Contractor
from pymps.mps import _autoset_bms
from blockmatrix import trunc_bm
from pydavidson import JDh
from tba.hgen import ind2c,kron_csr
from flib.flib import fget_subblock2a,fget_subblock2b,fget_subblock1

__all__=['VMPSEngine']

ZERO_REF=1e-12

def _eigsh(H,v0,projector=None,tol=1e-10,sigma=None,lc_search_space=1,k=1,iprint=0,which='SA',eigen_solver='LC'):
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
        overlap=abs(reshape(v0,[1,-1]).dot(V).ravel())
        ind=argmax(overlap)
        print 'Match Overlap = %s'%overlap[ind]
        return E[ind],V[:,ind:ind+1]
    if eigen_solver=='LC':
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
            e,v=JDh(H,v0=v0,k=k,projector=projector,tol=tol,maxiter=maxiter,linear_solver_maxiter=5,\
                    sigma=sigma,which='SA',iprint=iprint,linear_solver='gmres')
        else:
            if sigma is None:
                e,v=JDh(H,v0=v0,k=max(lc_search_space,k),projector=projector,tol=tol,maxiter=maxiter,\
                        linear_solver_maxiter=10,which='SA',iprint=iprint,linear_solver='gmres')
            else:
                e,v=JDh(H,v0=v0,k=k,projector=projector,tol=tol,sigma=sigma,which='SL',linear_solver_maxiter=5,\
                        iprint=iprint,converge_bound=1e-10,maxiter=maxiter,linear_solver='gmres')

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


class VMPSEngine(object):
    '''
    Variational MPS Engine.

    Attributes:
        :H: <MPO>, the hamiltonian operator.
        :ket: <MPS>, the eigen state.
        :labels: list, they are [physical-bra, physical-ket, bond-bra, bond-mpo, bond-ket]
        :nsite_update: int, the sites updated one time.
        :eigen_solver: str, eigenvalue solver.
            *'JD', Jacobi-Davidson method.
            *'LC', Lanczos, method.
        :iprint: int, print information level.
    '''
    def __init__(self,H,k0,labels=['s','m','a','b','c'],nsite_update=2,eigen_solver='JD',iprint=2):
        self.eigen_solver=eigen_solver
        self.nsite_update=nsite_update
        #set up initial ket
        ket=k0
        ket<<ket.l-1  #right normalize the ket to the first bond, where we start our update
        nsite=ket.nsite
        self.iprint=iprint

        #unify labels
        self.labels=labels
        ket.chlabel([labels[1],labels[4]])
        H.chlabel([labels[0],labels[1],labels[3]])

        #initial contractions,
        self.con=Contractor(H,ket,bra_bond_str=labels[2])
        self.con.initialize_env()

    @property
    def energy(self):
        '''Get the energy from current status.'''
        return self.con.evaluate().real

    @property
    def ket(self): return self.con.ket

    @property
    def H(self): return self.con.mpo

    def _get_iterator(self,start,stop):
        nsite=self.con.ket.nsite
        nsite_update=self.nsite_update

        #validate datas
        if start[1] not in ['->','<-'] or stop[1] not in ['->','<-']:
            raise ValueError()
        #check for null iterations
        if start[2]<0 or start[2]>nsite-nsite_update or stop[2]<0 or stop[2]>nsite-nsite_update:
            return
        if stop[0]<start[0]:
            return
        elif stop[0]==start[0]:
            if stop[1]=='->' and start[1]=='<-':
                return
            elif stop[0]==start[0] and stop[1]==start[1]:
                if start[1]=='->' and stop[2]<start[2]:
                    return
                elif start[1]=='<-' and stop[2]>start[2]:
                    return

        direction,site=start[1],start[2]
        for iiter in xrange(start[0],stop[0]+1):
            if self.iprint>1:
                print '########### STARTING NEW ITERATION %s ################'%(iiter)
            while(True):
                #if site>=0:  #to ensure 2 site update
                yield iiter,direction,site
                if iiter==stop[0] and direction==stop[1] and site==stop[2]:
                    return
                #next site and sweep direction
                if direction=='->':
                    if site>=nsite-nsite_update:
                        site-=1
                        direction='<-'
                    else:
                        site+=1
                else:
                    if site<=0:
                        site+=1
                        direction='->'
                        break
                    else:
                        site-=1
                if site<0:
                    direction,site='->',0
                    break
                if site>nsite-nsite_update:
                    direction,site='<-',nsite-nsite_update
                    break

    def run(self,nsweep,*args,**kwargs):
        self.sweep(start=(0,'->',0),stop=(nsweep-1,'<-',0),*args,**kwargs)

    def sweep(self,start,stop,maxN=50,tol=0,which='SA',iprint=1):
        '''
        Run this application.

        Parameters:
            :start: len-3 tuple, (the start sweep, moving direction, the start point).
            :stop: len-3 tuple, (the end sweep, moving direction, the end point).
            :maxN: list/int, the maximum kept dimension.
            :tol: float, the tolerence.
            :which: str, string that specify the <MPS> desired, 'SL'(most similar to k0) or 'SA'(smallest)

        Return:
            (E, <MPS>)
        '''
        #check data
        ket=self.ket
        nsite=ket.nsite
        hndim=ket.hndim
        nsite_update=self.nsite_update
        use_bm=hasattr(ket,'bmg')
        if use_bm: bmg=ket.bmg
        if isinstance(maxN,int): maxN=[maxN]*(stop[0]+1)
        iprint=self.iprint

        elist=[]
        iterator=get_sweeper(start,stop,nsite=nsite-nsite_update,iprint=self.iprint)
        for iiter,direction,l in iterator:
            self.con.initialize_env()
            if iprint>1:
                print 'Running iter = %s, direction = %s, l = %s'%(iiter+1,direction,l)
                print 'A'*(l)+'.'*nsite_update+'B'*(nsite-l-nsite_update)
            t0=time.time()

            #construct the Tensor for Hamilonian
            ket>>l+nsite_update/2-ket.l
            FL=self.con.LPART[l]
            FR=self.con.RPART[nsite-l-nsite_update]
            Os=[self.H.get(li) for li in xrange(l,l+nsite_update)]
            K0s=[ket.get(li,attach_S='B') for li in xrange(l,l+nsite_update)]
            t00=time.time()
            #get Tc,indices
            if use_bm:
                #get bmd = c(l-1)-m(l-1)-m(l)-c(l+1)
                bms=[lb.bm for lb in K0s[0].labels[:1]+[K.labels[1] for K in K0s]+K0s[-1].labels[-1:]]
                bmd,info=bmg.join_bms(bms,signs=[1]+[1]*nsite_update+[-1]).sort(return_info=True); pmd=info['pm']
                bmd=bmd.compact_form()
                #get the indices
                sls=bmd.get_slice(bmd.index_qn(zeros(len(bmg.qstring),dtype='int32')).item())
                indices=pmd[sls]
                #turn the indices into subindices.
                NN=array([bm.N for bm in bms])
                cinds=ind2c(indices,NN)
                #get the sub-block.
                if nsite_update==2:
                    TcL,TcR=(FL*Os[0]).chorder([0,2,1,3,4]),(Os[1]*FR).chorder([0,1,3,2,4]) #ascmb,bsamc kron-> assa,cmmc
                    TcL,TcR=TcL.reshape([prod(TcL.shape[:2]),-1,TcL.shape[-1]]),TcR.reshape([TcR.shape[0],prod(TcR.shape[1:3]),-1])
                elif nsite_update==1:
                    TcL,TcR=(FL*Os[0]).chorder([0,2,1,3,4]),FR.chorder([1,0,2]) #acsmb;bsmac
                    TcL=TcL.reshape([prod(TcL.shape[:2]),-1,TcL.shape[-1]])

                TcL.eliminate_zeros(ZERO_REF)
                TcR.eliminate_zeros(ZERO_REF)
                cdim=len(indices)
                Tc=csr_matrix((cdim,cdim))
                tk=tp=0
                for i in xrange(TcL.shape[-1]):
                    t00=time.time()
                    #TcL(as;mc;b), TcR(b;as;mc)
                    Tci=kron_csr(csr_matrix(TcL[:,:,i]),csr_matrix(TcR[i,:,:]),takerows=indices)
                    t11=time.time()
                    Tc=Tc+Tci.tocsc()[:,indices]
                    t22=time.time()
                    tk+=t11-t00
                    tp+=t22-t11
                if iprint>5:
                    print '@kron: %s, @sum: %s'%(tk,tp)
            else:
                if nsite_update==2:
                    Tc=(FL*Os[0]*Os[1]*FR).chorder([0,2,4,6,1,3,5,7])
                else:
                    Tc=(FL*Os[0]*FR).chorder([0,2,4,1,3,5])
                Tc.eliminate_zeros(ZERO_REF)
                Tc=csr_matrix(Tc.reshape([prod(Tc.shape[:2+nsite_update]),-1]))
            t1=time.time()

            #third, get the initial vector
            V0=reduce(lambda x,y:x*y,K0s)
            v0=asarray(V0.ravel())
            if use_bm:
                v0c=v0[indices]
            else:
                v0c=v0
            if which=='SA':
                E,Vc=_eigsh(Tc,v0=v0c,projector=None,tol=1e-10,sigma=None,lc_search_space=1,k=1,eigen_solver=self.eigen_solver)
            elif which=='SL':
                E,Vc=_eigsh(Tc.todense(),v0=v0c,which='SL',eigen_solver=self.eigen_solver)
            else:
                raise ValueError()

            if use_bm:
                V=zeros(bmd.N,dtype='complex128')
                V[indices]=Vc
            else:
                V=Vc
            overlap=v0.dot(V)
            t2=time.time()

            #update our ket,
            if nsite_update==2:
                #if 2 site update, perform svd and truncate.
                Vm=Tensor(V.reshape(K0s[0].shape[:2]+K0s[1].shape[1:]),labels=K0s[0].labels[:2]+K0s[1].labels[1:])
                K1,S,K2=Vm.svd(cbond=2,cbond_str='%s_%s'%(self.labels[-1],l+1),signs=[1,1,-1,1],bmg=bmg)
                #do the truncation
                if len(S)>maxN[iiter]:
                    Smin=sort(S)[-maxN[iiter]]
                    kpmask=S>=Smin
                    K1,S,K2=K1.take(kpmask,axis=-1),S[kpmask],K2.take(kpmask,axis=0)
                K1.eliminate_zeros(ZERO_REF)
                K2.eliminate_zeros(ZERO_REF)
                #set datas
                bdim=len(S)
                ket.ML[ket.l-1]=K1
                ket.S=S
                ket.ML[ket.l]=K2
            elif nsite_update==1:
                if direction=='->':
                    ket.ML[ket.l]=Tensor(V.reshape(V0.shape),labels=V0.labels)
                    ket.S=ones(V0.shape[0])  #S has been taken into consideration, so, don't use it anymore.
                    ket>>1
                else:
                    ket.l+=1
                    ket.ML[ket.l-1]=Tensor(V.reshape(V0.shape),labels=V0.labels)
                    ket.S=ones(V0.shape[-1])  #S has been taken into consideration, so, don't use it anymore.
                    ket<<1
                bdim=len(ket.S)

            #update our contractions.
            #1. the left part
            self.con.lupdate_env(ket.l)
            #2. the right part
            self.con.rupdate_env(nsite-ket.l)
            t3=time.time()

            #schedular check
            elist.append(E)
            diff=Inf if len(elist)<=1 else elist[-1]-elist[-2]
            if iprint>1:
                print 'Get E = %.12f, tol = %s, Elapse -> %s, %s states kept, nnz= %s, overlap %s.'%(E,'[-]' if diff==Inf else diff,t3-t0,bdim,Tc.nnz,overlap)
                print 'Time: get Tc(%s), eigen(%s), svd(%s)'%(t1-t0,t2-t1,t3-t2)
            if iiter==stop[0] and direction==stop[1] and l==stop[2]:
                if iprint>1:
                    print 'RUN COMPLETE!'
                return E,ket

            if direction=='<-' and l==0:
                #schedular check for each iteration
                if iiter==0:
                    E_last=Inf
                diff=E_last-E
                E_last=E
                if iprint>0:
                    print 'ITERATION SUMMARY: E/site = %s, tol = %s'%(E,'[-]' if diff==Inf else diff)

    def warmup(self,maxiter=10):
        '''Initialize the state.'''
        run10=run5=maxiter/3
        run20=maxiter-run10-run5
        self.run(maxiter,maxN=[3]*run5+[8]*run10+[16]*run20,which='SA')

    def generative_run(self,HP,ngen,niter_inner,S_pre=None,trunc_mps=False,*args,**kwargs):
        '''
        Parameters:
            :HP: list, mpo cells.
            :ngen: int, # of generations.
            :niter_inner: int, # of iteration for each generation.
            :S_pre: 1darray/None, the singular values of last iteration.
            :trunc_mps: bool, truncate MPS, used for infinite run.
        '''
        nsite_update=self.nsite_update
        ncell=len(HP)
        ket=self.con.ket
        iprint=self.iprint
        self.con.canomove(ket.nsite/2-ket.l)
        if S_pre is None: S_pre=ones(ket.check_link(ket.nsite/2-ncell/2))
        for i in xrange(ngen):
            l0=ket.nsite/2
            #make prediction of ket
            cells=[t.make_copy(copydata=False) for t in ket.ML[l0:l0+ncell/2]+ket.ML[l0-ncell/2:l0]]
            cells[0]=cells[0].mul_axis(ket.S,axis=0)
            cells[-1]=cells[-1].mul_axis(ket.S,axis=-1)
            ket.insert(l0,cells)
            ket.l=l0+ncell/2
            ket.S,S_pre=1/S_pre,ket.S
            #and canonicalize it
            self.con.ket>>ncell/2-nsite_update/2
            self.con.ket<<ncell-nsite_update
            self.con.ket>>ncell/2-nsite_update/2
            #update MPO
            self.con.mpo.insert(l0,[o.make_copy(copydata=False) for o in HP])
            if trunc_mps:
                self.con.keep_only(l0,ncell+l0)
                l0=0
            self.con.update_env_labels()
            #update environments
            for j in xrange(1,ncell/2):
                self.con.lupdate_env(l0+j)
                self.con.rupdate_env(l0+j)

            if iprint>0:
                print u'\u25cf EXPAND %s START'%(i)
            for iiter in xrange(niter_inner):
                start,stop=(iiter,'->',l0+ncell/2-nsite_update/2),(iiter,'->',l0+ncell-nsite_update)
                self.sweep(start,stop,*args,**kwargs)
                start,stop=(iiter,'<-',l0+ncell-nsite_update-1),(iiter,'<-',l0)
                self.sweep(start,stop,*args,**kwargs)
                start,stop=(iiter,'->',l0+1),(iiter,'->',l0+ncell/2-nsite_update/2)
                self.sweep(start,stop,*args,**kwargs)

            if iprint>0:
                print u'\u25cf EXPAND %s RES: Enery/Site = %s'%(i,self.con.evaluate()/(ket.nsite*(i+2 if trunc_mps else 1)))  #!

