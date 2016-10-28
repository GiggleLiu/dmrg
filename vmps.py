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

from rglib.mps import contract,Tensor,svdbd,check_validity_mps,BLabel,BMPS,check_flow_mpx,Contractor
from blockmatrix import trunc_bm
from pydavidson import JDh
from tba.hgen import ind2c
from rglib.hexpand import kron as skron2
from flib.flib import fget_subblock2a,fget_subblock2b,fget_subblock1,ftake_only

__all__=['VMPSEngine']

ZERO_REF=1e-12

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
        self.eigen_solver=eigen_solver
        #set up initial ket
        ket=k0
        ket<<ket.l-1  #right normalize the ket to the first bond, where we start our update
        nsite=ket.nsite

        #unify labels
        self.labels=labels
        ket.chlabel([labels[1],labels[4]])
        H.chlabel([labels[0],labels[1],labels[3]])

        #initial contractions,
        self.con=Contractor(H,ket,bra_bond_str=labels[2])
        self.con.contract2l()

    @property
    def energy(self):
        '''Get the energy from current status.'''
        return self.con.evaluate().real

    @property
    def ket(self): return self.con.ket

    @property
    def H(self): return self.con.mpo

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
            overlap=abs(reshape(v0,[1,-1]).dot(V).ravel())
            ind=argmax(overlap)
            print 'Match Overlap = %s'%overlap[ind]
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
            (E, <MPS>)
        '''
        #check data
        ket=self.ket
        maxiter=endpoint[0]+1
        nsite=ket.nsite
        hndim=ket.hndim
        use_bm=hasattr(ket,'bmg')
        if use_bm: bmg=ket.bmg
        if isinstance(maxN,int): maxN=[maxN]*maxiter
        if endpoint[1]=='->':
            assert(endpoint[2]>=0 and endpoint[2]<nsite-nsite_update)
        elif endpoint[1]=='<-':
            assert(endpoint[2]>0 and endpoint[2]<=nsite-nsite_update)
        else:
            raise ValueError()

        elist=[]
        for iiter in xrange(maxiter):
            print '########### STARTING NEW ITERATION %s ################'%(iiter+1)
            for direction,iterator in zip(['->','<-'],[xrange(nsite-nsite_update),xrange(nsite-nsite_update,0,-1)]):
                for l in iterator:   #l is the index of first site.
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
                        bmd,pmd=bmg.join_bms(bms,signs=[1]+[1]*nsite_update+[-1])
                        #get the indices taken
                        sls=bmd.get_slice(zeros(len(bmg.qstring),dtype='int32'),uselabel=True)
                        indices=pmd[sls]
                        #turn the indices into subindices.
                        NN=array([bm.N for bm in bms])
                        cinds=ind2c(indices,NN)
                        #get the sub-block.
                        if nsite_update==2:
                            #Tc=fget_subblock2b(FL,Os[0],Os[1],FR,cinds)
                            TcL,TcR=(FL*Os[0]).chorder([0,2,1,3,4]),(Os[1]*FR).chorder([0,1,3,2,4]) #acsmb;bsmac
                            TcL,TcR=TcL.reshape([prod(TcL.shape[:2]),-1,TcL.shape[-1]]),TcR.reshape([TcR.shape[0],prod(TcR.shape[1:3]),-1])
                            TcL[abs(TcL)<ZERO_REF]=0
                            TcR[abs(TcR)<ZERO_REF]=0
                            #Tc=coo_matrix((TcL.shape[0]*TcR.shape[1],TcL.shape[1]*TcR.shape[2]),dtype=TcL.dtype)
                            ta=time.time()
                            datas,rows,cols=[],[],[]
                            for i in xrange(TcL.shape[-1]):
                                tj=time.time()
                                Tci=skron2(csr_matrix(TcL[:,:,i]),csc_matrix(TcR[i,:,:]))
                                rowi=Tci.row
                                if len(rowi)>0:
                                    datai=Tci.data;coli=Tci.col
                                    mask1a,mask2b=ftake_only(rowi,coli,indices)
                                    rowi,ncoli,datai=rowi[mask1a.astype('bool')],coli[mask1a.astype('bool')]
                                    mask1b,mask2b=ftake_only(coli,indices)
                                    mask1,mask2=mask1a&mask1b,mask2a&mask2b
                                    rowi=where(mask2); coli=where(mask2); datai=datai[mask1]
                                    datas.append(datai); cols.append(coli); rows.append(rowi)
                                ti=time.time()
                                print ti-tj
                            dim=len(indices)
                            Tc=coo_matrix((concatenate(datas),(concatenate(rows),concatenate(cols))),shape=(dim,dim),dtype=TcL.dtype)
                            Tc=Tc.tocsr()#[indices][:,indices]
                            tb=time.time()
                            print '@%s'%(tb-ta)
                        elif nsite_update==1:
                            Tc=fget_subblock1(FL,Os[0],FR,cinds)
                    else:
                        if nsite_update==2:
                            Tc=(FL*Os[0]*Os[1]*FR).chorder([0,2,4,6,1,3,5,7])
                        else:
                            Tc=(FL*Os[0]*FR).chorder([0,2,4,1,3,5])
                        Tc=Tc.reshape([prod(Tc.shape[:2+nsite_update]),-1])
                    t1=time.time()

                    #third, get the initial vector
                    V0=reduce(lambda x,y:x*y,K0s)
                    v0=asarray(V0.ravel())
                    if use_bm:
                        v0c=v0[indices]
                    else:
                        v0c=v0
                    if which=='SA':
                        E,Vc=self._eigsh(Tc,v0=v0c,projector=None,tol=1e-10,sigma=None,lc_search_space=1,k=1)
                    elif which=='SL':
                        E,Vc=self._eigsh(Tc.todense(),v0=v0c,which='SL')
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
                        Vm=V.reshape([FL.shape[0]*hndim,FR.shape[0]*hndim])
                        if use_bm:
                            #first, find the block structure and make it block diagonal
                            bml=bmg.join_bms([V0.labels[0].bm,V0.labels[1].bm],signs=[1,1],compact_form=False)[0]
                            bmr=bmg.join_bms([V0.labels[2].bm,V0.labels[3].bm],signs=[-1,1],compact_form=False)[0]
                            K1K2=Tensor(Vm,labels=[BLabel('KL',bml),BLabel('KR',bmr)])
                            K1K2,pms=K1K2.b_reorder(return_pm=True)
                            #perform svd and roll back to original non-block structure
                            K1,S,K2=svdbd(K1K2,cbond_str='%s_%s'%(self.labels[-1],l+1))
                            K1,K2=K1[argsort(pms[0])],K2[:,argsort(pms[1])]
                        else:
                            #perform svd and roll back to original non-block structure
                            K1,S,K2=svd(Vm,full_matrices=False)
                            cbond_str='%s_%s'%(self.labels[-1],l+1)
                            K1=Tensor(K1,labels=['KL',cbond_str])
                            K2=Tensor(K2,labels=[cbond_str,'KR'])
                        #do the truncation
                        if len(S)>maxN[iiter]:
                            Smin=sort(S)[-maxN[iiter]]
                            kpmask=S>=Smin
                            K1,S,K2=K1[:,kpmask],S[kpmask],K2[kpmask]
                            if use_bm:
                                bm_new=trunc_bm(K2.labels[0].bm,kpmask=kpmask)
                                K2.labels[0].bm=K1.labels[1].bm=bm_new
                        K1[abs(K1)<ZERO_REF]=0
                        K2[abs(K2)<ZERO_REF]=0
                        #set datas
                        bdim=len(S)
                        ket.AL[-1]=Tensor(K1.reshape([K0s[0].shape[0],hndim,bdim]),labels=K0s[0].labels[:2]+K1.labels[-1:])
                        ket.S=S
                        ket.BL[0]=Tensor(K2.reshape([bdim,hndim,K0s[1].shape[2]]),labels=K2.labels[:1]+K0s[1].labels[1:])
                    elif nsite_update==1:
                        if direction=='->':
                            ket.BL[0]=Tensor(V.reshape(V0.shape),labels=V0.labels)
                            ket.S=ones(V0.shape[0])  #S has been taken into consideration, so, don't use it anymore.
                            ket>>1
                        else:
                            ket.AL.append(Tensor(V.reshape(V0.shape),labels=V0.labels)); ket.BL.pop(0)
                            ket.S=ones(V0.shape[-1])  #S has been taken into consideration, so, don't use it anymore.
                            ket<<1
                        bdim=len(ket.S)

                    #update our contractions.
                    #1. the left part
                    self.con.lupdate(ket.l)
                    #2. the right part
                    self.con.rupdate(nsite-ket.l)
                    t3=time.time()

                    #schedular check
                    elist.append(E)
                    diff=Inf if len(elist)<=1 else elist[-1]-elist[-2]
                    print 'Get E = %.12f, tol = %s, Elapse -> %s, %s states kept, overlap %s.'%(E,diff,t3-t0,bdim,overlap)
                    print 'Time: get Tc(%s), eigen(%s), svd(%s)'%(t1-t0,t2-t1,t3-t2)
                    if iiter+1==endpoint[0] and direction==endpoint[1] and l==endpoint[2]:
                        print 'RUN COMPLETE!'
                        return E,ket

            #schedular check for each iteration
            if iiter==0:
                E_last=Inf
            diff=E_last-E
            E_last=E
            print 'ITERATION SUMMARY: E/site = %s, tol = %s'%(E,diff)
