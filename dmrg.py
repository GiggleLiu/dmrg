'''
DMRG Engine.
'''

from numpy import *
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh,norm,svd
from matplotlib.pyplot import *
import scipy.sparse as sps
import copy,time,pdb,warnings,numbers

from blockmatrix.blocklib import eigbsh,eigbh,get_blockmarker
from rglib.mps import MPS,NORMAL_ORDER,SITE,LLINK,RLINK,chorder,OpString,tensor
from rglib.hexpand import NullEvolutor,Z4scfg,MaskedEvolutor,kron
from rglib.hexpand import signlib
from disc_symm import SymmetryHandler
from superblock import SuperBlock,site_image

ZERO_REF=1e-10

__all__=['site_image','SuperBlock','DMRGEngine']

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
            return self.LPART[length].make_copy()
        else:
            #return copy.deepcopy(self.RPART[length])
            return self.RPART[length].make_copy()

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
            tuple, the ground state energy and the ground state(in <MPS> form).
        '''
        EL=[]
        #check the validity of datas.
        if isinstance(self.hgen.evolutor,NullEvolutor):
            raise ValueError('The evolutor must not be null!')
        symm_handler=SymmetryHandler(dict(block_params.get('target_sector',{})),C_detect_scope=block_params.get('C_detect_scope',2))
        nlevel=block_params.get('nlevel',1)
        if not symm_handler.isnull and nlevel!=1:
            raise NotImplementedError('The symmetric Handler can not be used in multi-level calculation!')
        if not symm_handler.isnull and self.bmg is None:
            raise NotImplementedError('The symmetric Handler can not without Block marker generator!')

        nsite=self.hchain.nsite
        if endpoint is None: endpoint=(4,'<-',0)
        maxscan,end_direction,end_site=endpoint
        if ndim(maxN)==0:
            maxN=[maxN]*maxscan
        assert(len(maxN)>=maxscan and end_site<=(nsite-2 if not self.reflect else nsite/2-1))
        EG_PRE=Inf
        initial_state=None
        if self.reflect:
            iterators={'->':xrange(nsite/2),'<-':xrange(nsite/2-2,-1,-1)}
        else:
            iterators={'->':xrange(nsite-1),'<-':xrange(nsite-2,-1,-1)}
        for n,m in enumerate(maxN):
            for direction in ['->','<-']:
                for i in iterators[direction]:
                    print 'Running %s-th scan, iteration %s'%(n+1,i)
                    t0=time.time()
                    #setup generators and operators.
                    hgen_l=self.query('l',i)
                    if n==0 and direction=='->' and i<(nsite+1)/2:
                        hgen_r=hgen_l
                    else:
                        hgen_r=self.query('r',nsite-i-2)
                    print 'A'*hgen_l.N+'..'+'B'*hgen_r.N
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

                    #do state prediction
                    initial_state=None   #restore initial state.
                    phi=phil[0]
                    if nsite==nsite_true:
                        if self.reflect and nsite%2==0 and ((i==nsite/2-2 and direction=='->') or (i==nsite/2 and direction=='<-')):
                            initial_state=None
                        elif direction=='->' and i==nsite-2:  #for the case without reflection.
                            initial_state=phil[0].ravel()
                        elif direction=='<-' and i==0:
                            initial_state=phil[0].ravel()
                        else:
                            if self.reflect and direction=='->' and i==nsite/2-1:
                                direction='<-'
                                #phi=transpose(phi,[2,3,0,1])
                            initial_state=sum([self.state_prediction(phi,l=i+1,direction=direction) for phi in phil],axis=0)
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
                            return EG,self.get_mps(phi=phil[0],hgen_r=hgen_r,hgen_l=hgen_l,direction=direction)
                        #elif all(abs(diff)<tol):
                        #    print 'Breaking due to enough precision reached!'
                        #    return EG,self.get_mps(phi=phil[0],hgen_r=hgen_r,hgen_l=hgen_l,direction=direction)
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
        if isinstance(self.hgen.evolutor,NullEvolutor):
            raise ValueError('The evolutor must not be null!')
        symm_handler=SymmetryHandler(dict(block_params.get('target_sector',{})),C_detect_scope=block_params.get('C_detect_scope',3))
        nlevel=block_params.get('nlevel',1)
        if not symm_handler.isnull and nlevel!=1:
            raise NotImplementedError('The symmetric Handler can not be used in multi-level calculation!')
        if not symm_handler.isnull and self.bmg is None:
            raise NotImplementedError('The symmetric Handler can not without Block marker generator!')
        if not self.reflect and symm_handler.has_symmetry('C'):
            warnings.warn('Using reflection symmetry but no reflection, not reliable!!!!!!!!!!')

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
            EG,U,kpmask,err,phil=self.dmrg_step(hgen,hgen,opi,tol=tol,block_params=block_params,symm_handler=symm_handler)
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
        #filter operators to extract left-only and right-only blocks.
        for op in ops:
            siteindices=array(op.siteindex).reshape([1,-1])
            if any(siteindices>NL+NR+1) or any(siteindices<0):
                print 'Drop opstring %s'%op
            elif all(siteindices>NL):
                opstr=site_image(op,NL+1,NR+1)
                intraop_r.append(opstr)
            elif all(siteindices<=NL):
                intraop_l.append(op)
            else:
                interop.append(op)
        HL0=hgen_l.expand(intraop_l)
        if hgen_r is hgen_l:
            HR0=HL0
        else:
            HR0=hgen_r.expand(intraop_r)

        #blockize HL0 and HR0
        nlevel=block_params.get('nlevel',1)
        if self.bmg is not None:
            target_block=block_params.get('target_block')
            n=max(hgen_l.N,hgen_r.N)
            if isinstance(hgen_l.evolutor,MaskedEvolutor) and n>1:
                kpmask_l=hgen_l.evolutor.kpmask(hgen_l.N-2)
                kpmask_r=hgen_r.evolutor.kpmask(hgen_r.N-2)
            else:
                kpmask_l=kpmask_r=None
            bml=self.bmg.update_blockmarker(hgen_l.block_marker,kpmask=kpmask_l,nsite=hgen_l.N)
            bmr=self.bmg.update_blockmarker(hgen_r.block_marker,kpmask=kpmask_r,nsite=hgen_r.N)
        else:
            bml=None #get_blockmarker(HL0)
            bmr=None #get_blockmarker(HR0)

        H=kron(HL0,sps.identity(ndimr))+kron(sps.identity(ndiml),HR0)
        sb=SuperBlock(hgen_l,hgen_r)
        Hin=[]
        for op in interop:
            Hin.append(sb.get_op(op))
        H=H+sum(Hin)

        #get the starting initial eigen state!
        #(e,),v=eigsh(H,which='SA',k=1)
        t1=time.time()
        if initial_state is None:
            initial_state=random.random(H.shape[0])
        if not symm_handler.isnull:
            if hgen_l.N!=hgen_r.N or (not self.reflect and not (hgen_l is hgen_r)):  #forbidden using C2 symmetry at NL!=NR
            #if hgen_l is not hgen_r:
                symm_handler.update_handlers(useC=False)
            else:
                nl=bml.antiblockize(int32(1-signlib.get_sign_from_bm(bml,diag_only=True))/2)
                symm_handler.update_handlers(n=nl,useC=True)
            #v00=symm_handler.project_state(phi=initial_state)
            v00=initial_state
            #H=symm_handler.project_op(op=H)
            if hgen_l is hgen_r:
                assert(symm_handler.check_op(sum(Hin)))
        else:
            v00=initial_state

        #perform diagonalization
        ##first, detect specific block for diagonalization, get Hc and v0
        if self.bmg is None or target_block is None:
            Hc=H
            bm_tot=None
            v0=v00
            #e,v,bm_tot,H_bd=eigbsh(H,nsp=500,tol=tol*1e-2,which='S',maxiter=5000)
            #v=bm_tot.antiblockize(v).toarray()
            #vl=[v]
        else:
            if hasattr(target_block,'__call__'):
                target_block=target_block(nsite=hgen_l.N+hgen_r.N)
            bm_tot=self.bmg.add(bml,bmr,nsite=hgen_l.N+hgen_r.N)
            H_bd=bm_tot.blockize(H)

            Hc=bm_tot.lextract_block(H_bd,(target_block,target_block))
            v0=bm_tot.lextract_block(bm_tot.blockize(v00),(target_block,))

        ##second, diagonalize to get desired number of levels
        detect_C2=symm_handler.target_sector.has_key('C')# and not symm_handler.useC
        k=max(nlevel,symm_handler.C_detect_scope if detect_C2 else 1)
        #M=None# if len(symm_handler.symms)==0 else symm_handler.get_projector()
        if Hc.shape[0]<400:
            e,v=eigh(Hc.toarray())
            e,v=e[:k],v[:,:k]
        else:
            try:
                e,v=eigsh(Hc,k=k,which='SA',maxiter=5000,tol=tol*1e-2,v0=v0)
            except:
                e,v=eigsh(Hc,k=k+1,which='SA',maxiter=5000,tol=tol*1e-2,v0=v0)
        order=argsort(e)
        e,v=e[order],v[:,order]
        ###roll back blocks
        if bm_tot is not None:
            bindex=bm_tot.labels.index(target_block)
            vl=array([bm_tot.antiblockize(sps.coo_matrix((v[:,i],(arange(bm_tot.Nr[bindex],\
                    bm_tot.Nr[bindex+1]),zeros(len(v)))),shape=(bm_tot.N,1),dtype='complex128').toarray(),axes=(0,)).ravel()\
                    for i in xrange(v.shape[-1])])
        else:
            vl=v.T
        pdb.set_trace()
        overlaps=array([abs(v00.dot(vi.conj()))/norm(v00)/norm(vi) for vi in vl])
        if detect_C2 and symm_handler.useC:  #use symmetry projection to detect C2
            indices=symm_handler.locate(vl)
            print e
            print 'We find %s states meeting requirements, will take 1.'%len(indices)
            e,vl=array([e[indices[0]]]),[vl[indices[0]]]
        elif detect_C2:  #use state prediction to detect C2
            ind=argmax(overlaps)
            e,vl,overlaps=array([e[ind]]),[vl[ind]],[overlaps[ind]]
        print 'The goodness of the estimate -> %s'%(overlaps[0])
        t2=time.time()

        #do-wavefunction analysis
        for v in vl:
            v[abs(v)<ZERO_REF]=0
        U,spec,V=self.direct_analysis(phis=vl,bml=bml,bmr=bmr);V=V.T.conj()
        print len(vl)
        pdb.set_trace()
        #spec,U=self.rdm_analysis(phis=vl,bml=bml,bmr=bmr,side='l')
        #spec_2,V1=self.rdm_analysis(phis=vl,bml=bml,bmr=bmr,side='r')
        #pdb.set_trace()

        #do truncation
        kpmask=zeros(U.shape[1],dtype='bool')
        spec_cut=sort(spec)[max(0,len(kpmask)-maxN)]
        kpmask[(spec>=spec_cut)&(spec>ZERO_REF)]=True
        print '%s states kept.'%sum(kpmask)
        trunc_error=sum(spec[~kpmask])
        hgen_l.trunc(U=U,block_marker=bml,kpmask=kpmask)
        if not (hgen_l.N==hgen_r.N and self.reflect):
            hgen_r.trunc(U=V,block_marker=bmr,kpmask=kpmask)
        t3=time.time()
        print 'Elapse -> total:%s, eigen:%s'%(t3-t0,t2-t1)
        phil=[phi.reshape([ndiml/hndim,hndim,ndimr/hndim,hndim]) for phi in vl]
        return e,U,kpmask,trunc_error,phil

    def rdm_analysis(self,phis,bml,bmr,side):
        '''
        The analysis of reduced density matrix.
        
        Parameters:
            :phis: list of 1D array, the kept eigen states of current iteration.
            :bml/bmr: <BlockMarker>/int, the block marker for left and right blocks/or the dimensions.
            :side: 'l'/'r', view the left or right side as the system.

        Return:
            tuple of (spec, U), the spectrum and Unitary matrix from the density matrix.
        '''
        assert(side=='l' or side=='r')
        ndiml,ndimr=(bml,bmr) if isinstance(bml,numbers.Number) else (bml.N,bmr.N)
        phis=[phi.reshape([ndiml,ndimr]) for phi in phis]
        rho=0
        phil=[]
        if side=='l':
            for phi in phis:
                phi=sps.csr_matrix(phi)
                rho=rho+phi.dot(phi.T.conj())
                phil.append(phi)
            bm=bml
        else:
            for phi in phis:
                phi=sps.csc_matrix(phi)
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
                raise Exception('''Density matrix is not block diagonal, which is not expected,
        1. make sure your are using additive good quantum numbers.
        2. avoid ground state degeneracy.''')
        spec,U=eigbh(rho,bm=bm)
        print 'With %s(%s) blocks.'%(bm.nblock,bm.nblock)
        return spec,U

    def direct_analysis(self,phis,bml,bmr,tol=1e-8):
        '''
        The direct analysis of state(svd).
        
        Parameters:
            :phis: list of 1D array, the kept eigen states of current iteration.
            :bml/bmr: <BlockMarker>/int, the block marker for left and right blocks/or the dimensions.
            :tol: float, the maximum allowed truncation error.

        Return:
            tuple of (spec, U), the spectrum and Unitary matrix from the density matrix.
        '''
        if isinstance(bml,numbers.Number):
            use_bm=False
            ndiml,ndimr=bml,bmr
        else:
            ndiml,ndimr=bml.N,bmr.N
            use_bm=True
        phi=sum(phis,axis=0).reshape([ndiml,ndimr])/sqrt(len(phis))  #construct wave function of equal distribution of all states.
        phi[abs(phi)<ZERO_REF]=0
        if use_bm:
            phi=bml.blockize(phi,axes=(0,))
            phi=bmr.blockize(phi,axes=(1,))

        rhol=phi.dot(phi.T.conj())
        rhor=phi.T.conj().dot(phi)

        if bml is not None:
            if not bml.check_blockdiag(rhol,tol=1e-5) or not bmr.check_blockdiag(rhor,tol=1e-5):
                raise Exception('''Density matrix is not block diagonal, which is not expected,
        1. make sure your are using additive good quantum numbers.
        2. avoid ground state degeneracy.''')
        spec,U=eigbh(rhol,bm=bml,return_vecs=True)
        kpmask=spec>tol
        spec,U=spec[kpmask],U.tocsc()[:,kpmask]
        #using phi=USV to get V
        V=U.T.conj().dot(phi)/sqrt(spec[:,newaxis])   #V is row-wise othorgonal
        #spec2,V=eigbh(rhor,bm=bmr,return_vecs=True);V=V.T.conj()

        #check phi
        assert(allclose(V.dot(V.T.conj()),identity(V.shape[0])))
        phi2=U.dot(sps.diags(sqrt(spec),0)).dot(V)
        assert(sum(phi2*phi.conj())>0.9)
 
        return U,spec,V

    def state_prediction(self,phi,l,direction):
        '''
        Predict the state for the next iteration.

        Parameters:
            :phi: ndarray, the state from the last iteration, [llink, site1, rlink, site2]
            :l: int, the current division point, the size of left block.
            :direction: '->'/'<-', the moving direction.

        Return:
            ndarray, the new state in the basis |al+1,sl+2,sl+3,al+3>.

            reference -> PRL 77. 3633
        '''
        assert(direction=='<-' or direction=='->')
        phi=tensor.Tensor(phi,labels=['al','sl+1','al+2','sl+2']) #l=NL-1
        nsite=self.hchain.nsite
        NL,NR=l,nsite-l
        hgen_l,hgen_r=self.query('l',NL),self.query('r',NR)
        lr=NR-2 if direction=='->' else NR-1
        ll=NL-1 if direction=='->' else NL-2
        A=hgen_l.evolutor.A(ll,dense=True)   #get A[sNL](NL-1,NL)
        B=hgen_r.evolutor.A(lr,dense=True)   #get B[sNR](NL+1,NL+2)
        if direction=='->':
            A=tensor.Tensor(A,labels=['sl+1','al','al+1']).conj()
            B=tensor.Tensor(B,labels=['sl+3','al+3','al+2']).conj()    #!the conjugate?
            phi=tensor.contract([A,phi,B])
            phi=phi.chorder([0,1,3,2])
            if hasattr(hgen_r,'zstring'):  #cope with the sign problem
                n1=(1-Z4scfg(hgen_l.spaceconfig).diagonal())/2
                nr=(1-hgen_r.zstring[lr].diagonal())/2
                n_tot=n1[:,newaxis,newaxis]*(nr[:,newaxis]+n1)
                phi=phi*(1-2*(n_tot%2))
        else:
            A=tensor.Tensor(A,labels=['sl','al-1','al']).conj()
            B=tensor.Tensor(B,labels=['sl+2','al+2','al+1']).conj()    #!the conjugate?
            phi=tensor.contract([A,phi,B])
            phi=phi.chorder([1,0,3,2])
            if hasattr(hgen_r,'zstring'):  #cope with the sign problem
                n1=(1-Z4scfg(hgen_l.spaceconfig).diagonal())/2
                nr=(1-hgen_r.zstring[lr+1].diagonal())/2
                n_tot=n1*(nr[:,newaxis])
                phi=phi*(1-2*(n_tot%2))
        return phi

    def get_mps(self,phi,hgen_l,hgen_r,labels=['s','a'],direction=None):
        '''
        Get the MPS from run-time phi, and evolution matrices.

        Parameters:
            :phi: ndarray, the eigen-function of current step.
            :hgen_l/hgen_r: list, the hamiltonian generator for left/right block.
            :direction: '->'/'<-'/None, if None, the direction is provided by the truncation information.

        Return:
            <MPS>, the disired MPS, the canonicallity if decided by the current position.
        '''
        #get the direction
        if direction is None:
            direction='->' if hgen_l.truncated else '<-'
        assert(direction=='<-' or direction=='->')

        phi=tensor.Tensor(phi,labels=['al','sl+1','al+2','sl+2']) #l=NL-1
        NL,NR=hgen_l.N,hgen_r.N
        nsite=NL+NR
        if direction=='->':
            A=hgen_l.evolutor.A(NL-1,dense=True)   #get A[sNL](NL-1,NL)
            A=tensor.Tensor(A,labels=['sl+1','al','al+1\'']).conj()
            phi=tensor.contract([A,phi])
            phi=phi.chorder([0,2,1])   #now we get phi(al+1,sl+2,al+2)
            #decouple phi into S*B, B is column-wise othorgonal
            U,S,V=svd(phi.reshape([phi.shape[0],-1]),full_matrices=False)
            U=tensor.Tensor(U,labels=['al+1\'','al+1'])
            A=A*U  #get A(al,sl+1,al+1)
            B=transpose(V.reshape([S.shape[0],phi.shape[1],phi.shape[2]]),axes=(1,2,0))   #al+1,sl+2,al+2 -> sl+2,al+2,al+1, stored in column wise othorgonal format
            AL=hgen_l.evolutor.get_AL(dense=True)[:-1]+[A]
            BL=[B]+hgen_r.evolutor.get_AL(dense=True)[::-1]
        else:
            B=hgen_r.evolutor.A(NR-1,dense=True)   #get B[sNR](NL+1,NL+2)
            B=tensor.Tensor(B,labels=['sl+2','al+2','al+1\'']).conj()    #!the conjugate?
            phi=tensor.contract([phi,B])
            #decouple phi into A*S, A is row-wise othorgonal
            U,S,V=svd(phi.reshape([phi.shape[0]*phi.shape[1],-1]),full_matrices=False)
            V=tensor.Tensor(V,labels=['al+1','al+1\''])
            B=(V*B).chorder([1,2,0]).conj()   #al+1,sl+2,al+2 -> sl+2,al+2,al+1, for B is in transposed order by default.
            A=transpose(U.reshape([phi.shape[0],phi.shape[1],S.shape[0]]),axes=(1,0,2))   #al,sl+1,al+1 -> sl+1,al,al+1, stored in column wise othorgonal format
            AL=hgen_l.evolutor.get_AL(dense=True)+[A]
            BL=[B]+hgen_r.evolutor.get_AL(dense=True)[::-1][1:]
        AL=[chorder(ai,target_order=MPS.order,old_order=[SITE,LLINK,RLINK]).conj() for ai in AL]
        BL=[chorder(bi,target_order=MPS.order,old_order=[SITE,RLINK,LLINK]) for bi in BL]   #transpose
        mps=MPS(AL=AL,BL=BL,S=S,labels=labels)
        return mps
