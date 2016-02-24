'''
DMRG Engine.
'''

from numpy import *
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from matplotlib.pyplot import *
import scipy.sparse as sps
import copy,time,pdb,warnings

from blockmatrix.blocklib import eigbsh,eigbh,get_blockmarker
from rglib.mps import MPS,NORMAL_ORDER,SITE,LLINK,RLINK,chorder,OpString
from rglib.hexpand import NullEvolutor,Z4scfg,MaskedEvolutor,kron

ZERO_REF=1e-10

__all__=['site_image','SuperBlock','DMRGEngine']

def site_image(ops,NL,NR):
    '''
    Perform imaging transformation for operator sites.
    
    Parameters:
        :NL/NR: int, the number of sites in left and right block.
        :ops: list of <OpString>/<OpUnit>, the operator(s) for operation.

    Return:
        list of <OpString>/<OpUnit>, the operators after imaginary operation.
    '''
    opss=[]
    if not isinstance(ops,(list,tuple,ndarray)):
        ops=[ops]
        is_list=False
    else:
        is_list=True

    for opi in ops:
        oprl=[]
        if isinstance(opi,OpString):
            opunits=opi.opunits
        else:
            opunits=[opi]
        for ou in opunits:
            ou=copy.copy(ou)
            ou.siteindex=(NR-1)-(ou.siteindex-NL)
            oprl.append(ou)
        opi=prod(oprl)
        if isinstance(opi,OpString):
            opi.compactify()
        opss.append(opi)
    return opss if is_list else opss[0]

class SuperBlock(object):
    '''
    Super Block operation.
    
    Construct
    ------------------------
    SuperBlock(hl,hr)

    Attributes
    -----------------------
    hl/hr:
        Hamiltonian Generator for left and right blocks.
    '''
    def __init__(self,hl,hr):
        self.hl=hl
        self.hr=hr

    @property
    def nsite(self):
        '''Total number of sites'''
        return self.hl.N+self.hr.N

    @property
    def hndim(self):
        '''The hamiltonian dimension of a single site.'''
        return self.hl.hndim

    def get_op_onlink(self,ouA,ouB):
        '''
        Get the operator on the link.
        
        Parameters:
            :ouA/ouB: <OpUnit>, the opunit on left/right link site.

        Return:
            matrix, the hamiltonian term.
        '''
        NL,NR=self.hl.N,self.hr.N
        sgnr=self.hr.zstring.get(self.hr.N-1).diagonal()
        scfg=self.hl.spaceconfig
        assert(ouA.siteindex==NL-1 and ouB.siteindex==NR-1)
        if ouA.fermionic:
            assert(sgnr is not None)
            sgn=Z4scfg(scfg)
            mA=kron(sps.identity(self.hl.ndim/scfg.hndim),sps.csr_matrix(ouA.get_data()).dot(sgn))
            mB=kron(sps.diags(sgnr,0),sps.csr_matrix(ouB.get_data()))
        else:
            mA=kron(sps.identity(self.hl.ndim/scfg.hndim),sps.csr_matrix(ouA.data))
            mB=kron(sps.identity(self.hl.ndim/scfg.hndim),sps.csr_matrix(ouB.data))
        op=kron(mA,mB)
        return op

    def get_op(self,opstring):
        '''
        Get the hamiltonian from a opstring instance.

        Parameters:
            :opstring: <OpString>, the operator string.

        Return:
            matrix, the hamiltonian term.
        '''
        hndim=self.hndim
        siteindices=list(opstring.siteindex)
        nsite=self.nsite
        NL,NR=self.hl.N,self.hr.N
        if any(array(siteindices)>=self.nsite):
            raise ValueError('Site index out of range.')
        if not (len(siteindices)==len(unique(siteindices)) and all(diff(siteindices)>=0)):
            raise ValueError('Compact opstring is required!')
        opll,oprl=[],[]
        for i,ou in enumerate(opstring.opunits):
            if ou.siteindex<NL:
                opll.append(ou)
            else:
                ou=site_image(ou,NL,NR)
                oprl.append(ou)
        #handle the fermionic link.
        if len(opll)>0 and len(oprl)>0 and opll[0].fermionic:
            if len(opll)!=1 or opll[0].siteindex!=NL-1 or len(oprl)!=1 or oprl[0].siteindex!=NR-1:
                raise NotImplementedError('Only nearest neighbor term is allowed for fermionic links!')
            return self.get_op_onlink(opll[0],oprl[0])
        if len(opll)>0:
            opstr=prod(opll)
            if isinstance(opstr,OpString):
                opstr.compactify()
            opl=self.hl.get_newop(opstr)
        else:
            opl=sps.identity(self.hl.ndim)
        if len(oprl)>0:
            opstr=prod(oprl)
            if isinstance(opstr,OpString):
                opstr.compactify()
            opr=self.hr.get_newop(opstr)
        else:
            opr=sps.identity(self.hr.ndim)
        return kron(opl,opr)

class DMRGEngine(object):
    '''
    DMRG Engine.

    Attributes:
        :hchain: <OpCollection>, the chain hamiltonian.
        :hgen: <RGHGen>, hamiltonian Generator.
        :bmg: <BlockMarkerGenerator>, the block marker generator.
        :tol: float, the tolerence, when maxN and tol are both set, we keep the lower dimension.
        :symmetric: bool, True if left<->right symmetric, can be used to shortcut the run time.
        :_tails(private): list, the last item of A matrices, which is used to construct the <MPS>.
    '''
    def __init__(self,hchain,hgen,bmg=None,tol=0,symmetric=False):
        self.hchain=hchain
        self.tol=tol
        self.hgen=hgen
        self.bmg=bmg
        self._tails=None
        self.symmetric=symmetric
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
        if which=='l' or self.symmetric:
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
        if which=='l' or self.symmetric:
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
        hgen=copy.deepcopy(self.hgen)
        if isinstance(hgen.evolutor,NullEvolutor):
            raise ValueError('The evolutor must not be null!')
        nsite=self.hchain.nsite
        if endpoint is None: endpoint=(5,'->',nsite-2)
        maxscan,end_direction,end_site=endpoint
        if ndim(maxN)==0: maxN=[maxN]*maxscan
        assert(len(maxN)>=maxscan)
        EG_PRE=Inf
        for n,m in enumerate(maxN):
            for direction,iterator in zip(['->','<-'],[xrange(nsite-1),xrange(nsite-2,-1,-1)]):
                for i in iterator:
                    print 'Running %s-th scan, iteration %s'%(n,i)
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

                    #run a step
                    EG,U,kpmask,err,phil=self.dmrg_step(hgen_l,hgen_r,opi,direction=direction,tol=tol,maxN=m,block_params=block_params)
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
                        uu=U.tocsc()[:,kpmask]
                        ai=array([a.toarray() for a in hgen_l.evolutor.A(i)])
                        self._tails=[einsum('ijk,jil->lk',ai.conj(),phi) for phi in phil]  #A(osite,llink,rlink), phi(llink,osite,nsite)
                        self._tails=[[A[:,newaxis] for A in tail] for tail in self._tails]

                    EG=EG/(hgen_l.N+hgen_r.N)
                    if len(EL)>0:
                        diff=EG-EL[-1]
                    else:
                        diff=Inf
                    t1=time.time()
                    print 'EG = %s, dE = %s, Elapse -> %.4f(D=%s), TruncError -> %s'%(EG,diff,t1-t0,hgen.ndim,err)
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
            EG,U,kpmask,err,phil=self.dmrg_step(hgen,hgen,opi,tol=tol,block_params=block_params)
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

    def dmrg_step(self,hgen_l,hgen_r,ops,direction='->',tol=0,maxN=20,block_params={}):
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
        if self.bmg is not None:
            target_block=block_params.get('target_block')
            nlevel=block_params.get('nlevel',1)
            n=max(hgen_l.N,hgen_r.N)
            if isinstance(hgen_l.evolutor,MaskedEvolutor) and n>1:
                kpmask_l=hgen_l.evolutor.kpmask(hgen_l.N-2)
                kpmask_r=hgen_r.evolutor.kpmask(hgen_r.N-2)
            else:
                kpmask_l=kpmask_r=None
            bml=self.bmg.update_blockmarker(hgen_l.block_marker,kpmask=kpmask_l)
            bmr=self.bmg.update_blockmarker(hgen_r.block_marker,kpmask=kpmask_r)
        else:
            bml=get_blockmarker(HL0)
            bmr=get_blockmarker(HR0)

        H=kron(HL0,sps.identity(ndimr))+kron(sps.identity(ndiml),HR0)
        sb=SuperBlock(hgen_l,hgen_r)
        Hin=[]
        for op in interop:
            Hin.append(sb.get_op(op))
        H+=sum(Hin)

        #blockize and get the eigenvalues.
        #(e,),v=eigsh(H,which='SA',k=1)
        t1=time.time()
        if self.bmg is None or target_block is None:
            e,v,bm_tot,H_bd=eigbsh(H,nsp=500,tol=tol,which='S',maxiter=5000)
            v=bm_tot.antiblockize(v).toarray()
        else:
            if hasattr(target_block,'__call__'):
                target_block=target_block(nsite=hgen_l.N+hgen_r.N)
            bm_tot=self.bmg.add(bml,bmr)
            H_bd=bm_tot.blockize(H)
            Hc=bm_tot.lextract_block(H_bd,target_block)
            if Hc.shape[0]<400:
                e,v=eigh(Hc.toarray())
                e,v=e[:nlevel],v[:,:nlevel]
            else:
                e,v=eigsh(Hc,k=nlevel,which='SA',maxiter=5000,tol=tol)
                order=argsort(e)
                e,v=e[order],v[:,order]
            bindex=bm_tot.labels.index(target_block)
            vl=[bm_tot.antiblockize(sps.coo_matrix((v[:,i],(arange(bm_tot.Nr[bindex],\
                    bm_tot.Nr[bindex+1]),zeros(len(v)))),shape=(bm_tot.N,1),dtype='complex128')).toarray()\
                    for i in xrange(nlevel)]
        t2=time.time()
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
        kpmask[(spec>=spec_cut)&(spec>tol)]=True
        print '%s states kept.'%sum(kpmask)
        trunc_error=sum(spec[~kpmask])
        if direction=='->':
            hgen_l.trunc(U=U,block_marker=bm,kpmask=kpmask)
        else:
            hgen_r.trunc(U=U,block_marker=bm,kpmask=kpmask)
        t3=time.time()
        print 'Elapse -> total:%s, eigen:%s'%(t3-t0,t2-t1)
        phil=[phi.toarray().reshape([ndiml/hndim,hndim,ndimr]) for phi in phil]
        return e,U,kpmask,trunc_error,phil

    def direct_solve(self,n=None):
        '''
        Directly solve the ground state energy through lanczos.

        n:
            The length of change.
        '''
        nsite=self.hchain.nsite
        if n is None: n=nsite
        assert(n<=nsite and n>=0)
        hgen=copy.deepcopy(self.hgen)
        if not isinstance(hgen.evolutor,NullEvolutor):
            raise ValueError('The evolutor must be null!')

        for i in xrange(n):
            print 'Running iteration %s'%i
            t0=time.time()
            ops=self.hchain.query(i)
            intraop,interop=[],[]
            for op in ops:
                siteindices=array(op.siteindex)
                if any(siteindices>i):
                    interop.append(op)
                else:
                    intraop.append(op)
            hgen.expand(intraop)
            hgen.trunc()
            EG,EV=eigsh(hgen.H,k=1,which='SA')
            print 'Iteration %s, EG = %s'%(i,EG/hgen.N)
        return EG/self.nsite

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
        ML=[chorder(ai,target_order=MPS.order,old_order=[SITE,RLINK,LLINK]).conj() for ai in [tail]+hgen.evolutor.get_AL(dense=True)[::-1]]
        mps=MPS(AL=[],BL=ML,S=ones(1),labels=labels)
        return mps
