'''
DMRG Engine, with the normal order.
'''

from numpy import *
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh,norm
from matplotlib.pyplot import *
import scipy.sparse as sps
import copy,time,pdb,warnings

from blockmatrix.blocklib import eigbsh,eigbh
from rglib.mps import MPS,NORMAL_ORDER,SITE,LLINK,RLINK,chorder,OpString,tensor
from rglib.hexpand import NullEvolutor,Z4scfg,MaskedEvolutor,kron
from rglib.hexpand import signlib
from disc_symm import SymmetryHandler

ZERO_REF=1e-10

__all__=['SuperBlock','DMRGEngine']

class DMRGEngine(object):
    '''
    DMRG Engine.

    Attributes:
        :hchain: <OpCollection>, the chain hamiltonian.
        :hgen: <RGHGen>, hamiltonian Generator.
        :bmg: <BlockMarkerGenerator>, the block marker generator.
        :tol: float, the tolerence, when maxN and tol are both set, we keep the lower dimension.
        :reflect: bool, True if left<->right reflect, can be used to shortcut the run time.
        :LPART/RPART: dict, the left/right scanning of hamiltonian generators.
        :_tails(private): list, the last item of A matrices, which is used to construct the <MPS>.
    '''
    def __init__(self,hchain,hgen,bmg=None,tol=0,reflect=False):
        self.hchain=hchain
        self.tol=tol
        self.hgen=hgen
        self.bmg=bmg
        self.reflect=reflect

        #claim attributes with dummy values.
        self._tails=None
        self.LPART=None
        self.RPART=None
        self.reset()

    @property
    def nsite(self):
        '''Number of sites'''
        return self.hchain.nsite

    def query(self,which,length):
        '''
        Query the hamiltonian generator of specific part.

        which:
            `l` -> the left part.
            `r` -> the right part.
        length:
            The length of block.
        '''
        assert(which=='l' or which=='r')
        if which=='l' or self.reflect:
            #return copy.deepcopy(self.LPART[length])
            return copy.copy(self.LPART[length])
        else:
            #return copy.deepcopy(self.RPART[length])
            return copy.copy(self.RPART[length])

    def set(self,which,hgen,length=None):
        '''
        Set the hamiltonian generator for specific part.

        Parameters:
            :which: str,

                * `l` -> the left part.
                * `r` -> the right part.
            :hgen: <RGHGen>, the RG hamiltonian generator.
            :length: int, the length of block, if set, it will do a length check.
        '''
        assert(length is None or length==hgen.N)
        assert(hgen.truncated)
        if which=='l' or self.reflect:
            self.LPART[hgen.N]=hgen
        else:
            self.RPART[hgen.N]=hgen

    def reset(self):
        '''Restore this engine to initial status.'''
        hgen=copy.deepcopy(self.hgen)
        self.LPART={0:hgen}
        self.RPART={0:hgen}

    def run_finite(self,endpoint=None,tol=0,maxN=20,block_params={}):
        '''
        Run the application.

        Parameters:
            :endpoint: tuple, the end position tuple of (scan, direction, size of left-block).
            :tol: float, the rolerence of energy.
            :maxN: int, maximum number of kept states and the tolerence for truncation weight.
            :block_params: dict, the parameters for block specification, key words:

                * target_block, the target block to evaluate the ground state energy.

        Return:
            list, the ground state energy of each scan.
        '''
        EL=[]
        symm_handler=SymmetryHandler(dict(block_params.get('target_sector',{})))
        if isinstance(self.hgen.evolutor,NullEvolutor):
            raise ValueError('The evolutor must not be null!')
        nsite=self.hchain.nsite
        if endpoint is None: endpoint=(5,'->',nsite-2)
        maxscan,end_direction,end_site=endpoint
        if ndim(maxN)==0: maxN=[maxN]*maxscan
        assert(len(maxN)>=maxscan)
        EG_PRE=Inf
        initial_state=None
        for n,m in enumerate(maxN):
            for direction,iterator in zip(['->','<-'],[xrange(nsite-1),xrange(nsite-2,-1,-1)]):
                for i in iterator:
                    print 'Running %s-th scan, iteration %s'%(n,i)
                    t0=time.time()
                    #setup generators and operators.
                    hgen_l=self.query('l',i)
                    if n==0 and direction=='->' and i<(nsite+1)/2:
                        hgen_r=copy.copy(hgen_l)
                    else:
                        hgen_r=copy.copy(query('r',nsite-i-2))
                    print 'A'*hgen_l.N+'..'+'B'*hgen_r.N
                    print hgen_l.ndim,hgen_r.ndim
                    opi=set(self.hchain.query(i)+self.hchain.query(i+1))
                    opi=filter(lambda op:all(array(op.siteindex)<=(hgen_l.N+hgen_r.N+1)),opi)
                    nsite_true=hgen_l.N+hgen_r.N+2

                    #run a step
                    EG,U,kpmask,err,phil=self.dmrg_step(hgen_l,hgen_r,opi,direction=direction,tol=tol,maxN=m,block_params=block_params,initial_state=initial_state,symm_handler=symm_handler)
                    #update LPART and RPART
                    if direction=='->':
                        self.set('l',hgen_l,i+1)
                        print 'setting %s(%s)-site of left'%(i+1,hgen_l.N)
                        if n==0 and i<(nsite-1)/2:
                            print 'setting %s(%s)-site of right'%(i+1,hgen_l.N)
                            self.set('r',hgen_l,i+1)
                    else:
                        print 'setting %s(%s)-site of right'%(nsite-i-1,hgen_r.N)
                        self.set('r',hgen_r,nsite-i-1)

                    if direction=='->' and i==nsite-2:  #fix tails
                        self.update_tail(phil[0],nsite=nsite)

                    #do state prediction
                    initial_state=None   #restore initial state.
                    if nsite==nsite_true:
                        if self.reflect and nsite%2==0 and ((i==nsite/2-2 and direction=='->') or (i==nsite/2 and direction=='<-')):
                            initial_state=None
                        elif direction=='->' and i==nsite-2:
                            initial_state=phil[0].ravel()
                        elif direction=='<-' and i==0:
                            initial_state=phil[0].ravel()
                        else:
                            initial_state=sum([self.state_prediction(phi,hgen_l=hgen_l,hgen_r=hgen_r,direction=direction) for phi in phil],axis=0)
                            initial_state=initial_state.ravel()

                    EG=EG/nsite_true
                    if len(EL)>0:
                        diff=EG-EL[-1]
                    else:
                        diff=Inf
                    t1=time.time()
                    print 'EG = %s, dE = %s, Elapse -> %.4f, TruncError -> %s'%(EG,diff,t1-t0,err)
                    EL.append(EG)
                    if i==end_site and direction==end_direction:
                        diff=EG-EG_PRE
                        print 'MidPoint -> EG = %s, dE = %s'%(EG,diff)
                        if n==maxscan-1:
                            print 'Breaking due to maximum scan reached!'
                            return EL
                        elif all(abs(diff)<tol):
                            print 'Breaking due to enough precision reached!'
                            return EL
                        else:
                            EG_PRE=EG

    def run_infinite(self,maxiter=50,tol=0,maxN=20,block_params={}):
        '''
        Run the application.

        maxiter:
            The maximum iteration times.
        tol:
            The rolerence of energy.
        maxN:
            Maximum number of kept states and the tolerence for truncation weight.
        '''
        EL=[]
        hgen=copy.deepcopy(self.hgen)
        if isinstance(hgen.evolutor,NullEvolutor):
            raise ValueError('The evolutor must not be null!')
        if maxiter>self.hchain.nsite:
            warnings.warn('Max iteration exceeded the chain length!')
        for i in xrange(maxiter):
            print 'Running iteration %s'%i
            t0=time.time()
            opi=unique(self.hchain.query(i)+self.hchain.query(i+1))
            opi=filter(lambda op:all(array(op.siteindex)<=(2*hgen.N+1)),opi)
            EG,U,kpmask,err,phil=self.dmrg_step(hgen,copy.copy(hgen),opi,tol=tol,block_params=block_params)
            EG=EG/(2.*(i+1))
            if len(EL)>0:
                diff=EG-EL[-1]
            else:
                diff=Inf
            t1=time.time()
            print 'EG = %.5f, dE = %.2e, Elapse -> %.4f(D=%s), TruncError -> %.2e'%(EG,diff,t1-t0,hgen.ndim,err)
            EL.append(EG)
            if abs(diff)<tol:
                print 'Breaking!'
                break
        return EL

    def dmrg_step(self,hgen_l,hgen_r,ops,direction='->',tol=0,maxN=20,block_params={},initial_state=None,symm_handler=None):
        '''
        Run a single step of DMRG iteration.

        Parameters:
            :hgen_l,hgen_r: <RGHGen>, the hamiltonian generator for left and right blocks.
            :ops: list of <OpString>/<OpUnit>, the relevant operators to update.
            :direction: str,

                * '->', right scan.
                * '<-', left scan.
            :tol: float, the rolerence.
            :maxN: int, maximum number of kept states and the tolerence for truncation weight.
            :initial_state: 1D array/None, the initial state(prediction), None for random.
            :symm_handler: <SymmetryHandler>, the symmetry handler instance.

        Return:
            tuple of (ground state energy(float), unitary matrix(2D array), kpmask(1D array of bool), truncation error(float))
        '''
        assert(direction=='->' or direction=='<-')
        t0=time.time()
        intraop_l,intraop_r,interop=[],[],[]
        hndim=hgen_l.hndim
        NL,NR=hgen_l.N,hgen_r.N
        ndiml0,ndimr0=hgen_l.ndim,hgen_r.ndim
        ndiml,ndimr=ndiml0*hndim,ndimr0*hndim
        #for op in ops:
        #    siteindices=array(op.siteindex).reshape([1,-1])
        #    if any(siteindices>NL+NR+1) or any(siteindices<0):
        #        print 'Drop opstring %s'%op
        #    elif all(siteindices>NL):
        #        opstr=sb.site_image(op)
        #        intraop_r.append(opstr)
        #    elif all(siteindices<=NL):
        #        intraop_l.append(op)
        #    else:
        #        interop.append(op)
        HL0=hgen_l.H#expand(intraop_l)
        HR0=hgen_r.H
        #blockize HL0 and HR0
        if self.bmg is not None:
            target_block=block_params.get('target_block')
            nlevel=block_params.get('nlevel',1)
            n=max(hgen_l.N,hgen_r.N)
            if isinstance(hgen_l.evolutor,MaskedEvolutor) and n>1:
                kpmask_l=hgen_l.evolutor.kpmask(hgen_l.N-2)
                kpmask_r=hgen_r.evolutor.kpmask(hgen_r.N-2)
            else:
                kpmask_l=kpmask_r=None
            bml=self.bmg.update_blockmarker(hgen_l.block_marker,kpmask=kpmask_l,nsite=hgen_l.N)
            bmr=self.bmg.update_blockmarker(hgen_r.block_marker,kpmask=kpmask_r,nsite=hgen_r.N)
        else:
            bml=bmr=None

        H=kron(HL0,sps.identity(hndim**2*hgen_r.ndim))+kron(sps.identity(hgen_l.ndim*hndim**2),HR0)
        sb=SuperBlock(hgen_l,hgen_r)
        #Hin=[]
        #for op in interop+sb.site_image(intraop_r):
        #    Hin.append(sb.get_op(op))
        #H0=H
        #H=H+sum(Hin)
        HL=[]
        for op in ops:
            HL.append(sb.get_op(op))
        H=H+sum(HL)

        #get the starting initial eigen state!
        #(e,),v=eigsh(H,which='SA',k=1)
        t1=time.time()
        if initial_state is None:
            initial_state=random.random(H.shape[0])
        if symm_handler is not None:
            if hgen_l.N!=hgen_r.N or not self.reflect:  #forbidden using C2 symmetry at NL!=NR
                symm_handler.update_handlers(useC=False)
                v00=initial_state
            else:
                nl=bml.antiblockize(int32(1-signlib.get_sign_from_bm(bml,diag_only=True))/2)
                symm_handler.update_handlers(n=nl,useC=True)
                v00=symm_handler.project_state(phi=initial_state)
                #H=symm_handler.project_hamiltonian(H=H)
                assert(symm_handler.check_op(H))
        else:
            v00=initial_state

        if self.bmg is None or target_block is None:
            e,v,bm_tot,H_bd=eigbsh(H,nsp=500,tol=tol*1e-2,which='S',maxiter=5000)
            v=bm_tot.antiblockize(v).toarray()
            vl=[v]
        else:
            if hasattr(target_block,'__call__'):
                target_block=target_block(nsite=hgen_l.N+hgen_r.N)
            bm_tot=self.bmg.add(bml,bmr,nsite=hgen_l.N+hgen_r.N)
            H_bd=bm_tot.blockize(H)

            Hc=bm_tot.lextract_block(H_bd,target_block)
            v0=bm_tot.lextract_block(bm_tot.blockize(v00),target_block)
            projector=symm_handler.get_projector()
            try:
                e,v=eigsh(Hc,k=nlevel,which='SA',maxiter=5000,tol=tol*1e-2,v0=v0,M=M)
            except:
                e,v=eigsh(Hc,k=nlevel+1,which='SA',maxiter=5000,tol=tol,v0=v0)
                e,v=e[:nlevel],v[:,:nlevel]
            order=argsort(e)
            e,v=e[order],v[:,order]
            bindex=bm_tot.labels.index(target_block)
            vl=[bm_tot.antiblockize(sps.coo_matrix((v[:,i],(arange(bm_tot.Nr[bindex],\
                    bm_tot.Nr[bindex+1]),zeros(len(v)))),shape=(bm_tot.N,1),dtype='complex128')).toarray()\
                    for i in xrange(nlevel)]
            if hgen_r.N==hgen_l.N and self.reflect and symm_handler.has_symmetry('C'):
                comps0=symm_handler.handlers['C'].check_parity(v00)
                comps=symm_handler.handlers['C'].check_parity(vl[0][:,0])
                print comps0,comps
        t2=time.time()
        v00=v00/norm(v00)
        vl0=vl[0][:,0]
        print 'The goodness of the estimate -> %s'%(abs(v00.dot(vl0.conj()))**2)
        vl=[v.reshape([ndiml,ndimr]) for v in vl]
        for v in vl:
            v[abs(v)<ZERO_REF]=0
        rho=0
        phil=[]
        if direction=='->':
            for v in vl:
                phi=sps.csr_matrix(v)
                rho=rho+phi.dot(phi.T.conj())
                phil.append(phi)
            bm=bml
        else:
            for v in vl:
                phi=sps.csc_matrix(v)
                rho=rho+phi.T.dot(phi.conj())
                phil.append(phi)
            bm=bmr
        if bm is not None:
            rho=bm.blockize(rho)
            if not bm.check_blockdiag(rho,tol=1e-5):
                ion()
                pcolor(exp(abs(rho.toarray().real)))
                bm.show()
                pdb.set_trace()
                raise Exception('density matrix is not block diagonal, which is not expected, make sure your are using additive good quantum numbers.')
        spec,U,bm,rho_b=eigbh(rho,bm=bm)
        print 'Find %s(%s) blocks.'%(bm.nblock,bm_tot.nblock)

        kpmask=zeros(U.shape[1],dtype='bool')
        spec_cut=sort(spec)[max(0,len(kpmask)-maxN)]
        kpmask[(spec>=spec_cut)&(spec>tol*1e-2)]=True
        print '%s states kept.'%sum(kpmask)
        trunc_error=sum(spec[~kpmask])
        if direction=='->':
            hgen_l.trunc(U=U,block_marker=bm,kpmask=kpmask)
        else:
            hgen_r.trunc(U=U,block_marker=bm,kpmask=kpmask)
        t3=time.time()
        print 'Elapse -> total:%s, eigen:%s'%(t3-t0,t2-t1)
        phil=[phi.toarray().reshape([ndiml/hndim,hndim,hndim,ndimr/hndim]) for phi in phil]
        return e,U,kpmask,trunc_error,phil

    def state_prediction(self,phi,hgen_l,hgen_r,direction):
        '''
        Predict the state for the next iteration.

        Parameters:
            :phi: ndarray, the state from the last iteration, [llink, site1, rlink, site2]
            :hgen_l/hgen_r: <RGHGen>, the hamiltonian generator for the left/right block.
            :direction: '->'/'<-', the moving direction.

        Return:
            ndarray, the new state in the basis |al+1,sl+2,sl+3,al+3>.

            reference -> PRL 77. 3633
        '''
        assert(direction=='<-' or direction=='->')
        phi=tensor.Tensor(phi,labels=['al','sl+1','sl+2','al+2']) #l=NL-1
        NL,NR=hgen_l.N,hgen_r.N
        nsite=NL+NR
        lr=NR-2 if direction=='->' else NR-1
        ll=NL-1 if direction=='->' else NL-2
        A=hgen_l.evolutor.A(ll,dense=True)   #get A[sNL](NL-1,NL)
        B=hgen_r.evolutor.A(lr,dense=True)   #get B[sNR](NL+1,NL+2)
        if direction=='->':
            A=tensor.Tensor(A,labels=['sl+1','al','al+1']).conj()
            B=tensor.Tensor(B,labels=['sl+3','al+3','al+2']).conj()    #!the conjugate?
            phi=tensor.contract([A,phi,B])
            phi=phi.chorder([0,1,2,3])
        else:
            A=tensor.Tensor(A,labels=['sl','al-1','al']).conj()
            B=tensor.Tensor(B,labels=['sl+2','al+2','al+1']).conj()    #!the conjugate?
            phi=tensor.contract([A,phi,B])
            phi=phi.chorder([1,0,2,3])
        return phi

    def update_tail(self,phi,nsite):
        '''
        update the last transform matrix.
        '''
        hgen_l=self.query('l',nsite-1)
        phi=tensor.Tensor(phi,labels=['al','sl+1','al+2','sl+2']) #l=NL-1
        A=hgen_l.evolutor.A(nsite-2,dense=True)
        A=tensor.Tensor(A,labels=['sl+1','al','al+1']).conj()
        phi=tensor.contract([A,phi])
        phi=phi.chorder([2,0,1])
        self._tails=[phi]
        #self._tails=[einsum('ijk,jiml->lkm',A.conj(),phi) for phi in phil]  #A(osite,llink,rlink), phi(llink,osite,nsite)

    def get_mps(self,labels=('s','a'),target_level=0):
        '''
        Transform <Evolutor> instance to <MPS> instance.

        Parameters:
            :labels: tuple of char,
                (label_site,label_link), The labels for degree of freedom on site and intersite links.
            :target_level: int, the n-th lowest level, 0 for ground state.

        Return:
            <MPS>, the desired matrix product state.

            Note: this mps is right canonical.
        '''
        hgen=self.query('l',self.hchain.nsite-1)
        tail=self._tails[target_level]
        #ML=[chorder(ai,target_order=MPS.order,old_order=[SITE,RLINK,LLINK]).conj() for ai in [tail]+hgen.evolutor.get_AL(dense=True)[::-1]]
        #mps=MPS(AL=[],BL=ML,S=ones(1),labels=labels)
        ML=[chorder(ai,target_order=MPS.order,old_order=[SITE,LLINK,RLINK]) for ai in hgen.evolutor.get_AL(dense=True)+[tail]]
        mps=MPS(AL=ML,BL=[],S=ones(1),labels=labels)
        return mps
