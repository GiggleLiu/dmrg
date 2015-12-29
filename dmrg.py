'''
DMRG Engine.
'''

from numpy import *
from scipy.sparse.linalg import eigsh
import scipy.sparse as sps
from scipy.linalg import eigh
import copy,time,pdb,warnings

from blockmatrix.blocklib import eigbsh,eigbh
from rglib.mps import MPS,NORMAL_ORDER,SITE,LLINK,RLINK,chorder,OpString
from rglib.hexpand import NullEvolutor

ZERO_REF=1e-12

__all__=['site_image','SuperBlock','DMRGEngine']

def site_image(ops,NL,NR):
    '''
    Perform imaging transformation for operator sites.
    
    NL/NR:
        The number of sites in left and right block.
    ops:
        The operator(s) for operation.
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

    def get_op(self,opstring):
        '''
        Get the hamiltonian from a opstring instance.

        opstring:
            The hamiltonian string.
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
                #ou=copy.copy(ou)
                #ou.siteindex=(nsite-NL-1)-(ou.siteindex-NL)
                ou=site_image(ou,NL,NR)
                oprl.append(ou)
        if len(opll)>0:
            opstr=prod(opll)
            if isinstance(opstr,OpString):
                opstr.compactify()
            opl=self.hl.get_op(opstr)
        else:
            opl=sps.identity(self.hl.ndim)
        if len(oprl)>0:
            opstr=prod(oprl)
            if isinstance(opstr,OpString):
                opstr.compactify()
            opr=self.hr.get_op(opstr)
        else:
            opr=sps.identity(self.hr.ndim)
        return sps.kron(opl,opr)

class DMRGEngine(object):
    '''
    DMRG Engine.

    Attributes
    --------------------------
    hchain:
        A chain of hamiltonian, an <OpCollection> instance.
    tol:
        When maxN and tol are both set, we keep the lower dimension.
    hgen:
        Hamiltonian Generator.
    maxbond:
        The maximum bond length.
    '''
    def __init__(self,hchain,hgen,tol=0):
        self.hchain=hchain
        self.tol=tol
        self.hgen=hgen
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
        if which=='l':
            return copy.deepcopy(self.LPART[length])
        else:
            return copy.deepcopy(self.RPART[length])

    def set(self,which,hgen,length=None):
        '''
        Set the hamiltonian generator for specific part.

        which:
            `l` -> the left part.
            `r` -> the right part.
        hgen:
            The hamiltonian generator.
        length:
            The length of block, if set, it will do a length check.
        '''
        assert(length is None or length==hgen.N)
        assert(hgen.truncated)
        if which=='l':
            self.LPART[hgen.N]=hgen
        else:
            self.RPART[hgen.N]=hgen

    def reset(self):
        '''Restore this engine to initial status.'''
        hgen=copy.deepcopy(self.hgen)
        self.LPART={0:hgen}
        self.RPART={0:hgen}

    def run_finite(self,endpoint=None,tol=0,maxN=20):
        '''
        Run the application.

        endpoint:
            The end position tuple of (scan, direction, size of left-block).
        tol:
            The rolerence of energy.
        maxN:
            Maximum number of kept states and the tolerence for truncation weight.
        '''
        EL=[]
        hgen=copy.deepcopy(self.hgen)
        if isinstance(hgen.evolutor,NullEvolutor):
            raise ValueError('The evolutor must not be null!')
        nsite=self.hchain.nsite
        if endpoint is None: endpoint=(5,'<-',nsite/2)
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
                    opi=unique(self.hchain.query(i)+self.hchain.query(i+1))
                    opi=filter(lambda op:all(array(op.siteindex)<=(hgen_l.N+hgen_r.N+1)),opi)

                    #run a step
                    EG,U,kpmask,err=self.dmrg_step(hgen_l,hgen_r,opi,direction=direction,tol=tol,maxN=m)
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

                    EG=EG/(hgen_l.N+hgen_r.N)
                    if len(EL)>0:
                        diff=EG-EL[-1]
                    else:
                        diff=Inf
                    t1=time.time()
                    print 'EG = %.10f, dE = %.2e, Elapse -> %.4f(D=%s), TruncError -> %.2e'%(EG,diff,t1-t0,hgen.ndim,err)
                    EL.append(EG)
                    if i==end_site and direction==end_direction:
                        diff=EG-EG_PRE
                        print 'MidPoint -> EG = %.10f, dE = %.2e'%(EG,diff)
                        if n==maxscan-1:
                            print 'Breaking due to maximum scan reached!'
                            return EL
                        elif abs(diff)<tol:
                            print 'Breaking due to enough precision reached!'
                            return EL
                        else:
                            EG_PRE=EG

    def run_infinite(self,maxiter=50,tol=0,maxN=20):
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
            EG,U,kpmask,err=self.dmrg_step(hgen,hgen,opi,tol=tol)
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

    def dmrg_step(self,hgen_l,hgen_r,ops,direction='->',tol=0,maxN=20):
        '''
        Run a single step of DMRG iteration.

        hgen:
            The hamiltonian generator.
        ops:
            The links between operators.
        direction:
            '->', right scan.
            '<-', left scan.
        maxN:
            Maximum number of kept states and the tolerence for truncation weight.
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
                #oprl=[]
                #for ou in op.opunits:
                #    ou=copy.copy(ou)
                #    ou.siteindex=NR-(ou.siteindex-NL-1)
                #    oprl.append(ou)
                #opstr=prod(oprl)
                #if isinstance(opstr,OpString):
                #    opstr.compactify()
                opstr=site_image(op,NL+1,NR+1)
                intraop_r.append(opstr)
            elif all(siteindices<=NL):
                intraop_l.append(op)
            else:
                interop.append(op)
        #print 'LOP:',intraop_l
        #print 'ROP',intraop_r
        #print 'INTER',interop
        HL0=hgen_l.expand(intraop_l)
        if hgen_r is hgen_l:
            HR0=HL0
        else:
            HR0=hgen_r.expand(intraop_r)
        H=sps.kron(HL0,sps.identity(ndimr))+sps.kron(sps.identity(ndiml),HR0)
        sb=SuperBlock(hgen_l,hgen_r)
        Hin=[]
        for op in interop:
            Hin.append(sb.get_op(op))
        H+=sum(Hin)

        #blockize and get the eigenvalues.
        #(e,),v=eigsh(H,which='SA',k=1)
        t1=time.time()
        e,v,bm,H=eigbsh(H,nsp=500,tol=1e-10,maxiter=5000)
        v=bm.antiblockize(v).toarray()
        t2=time.time()
        v=v.reshape([ndiml,ndimr])
        v[abs(v)<ZERO_REF]=0
        if direction=='->':
            phi=sps.csr_matrix(v)
            rho=phi.dot(phi.T.conj())
        else:
            phi=sps.csc_matrix(v)
            rho=phi.T.dot(phi.conj())
        spec,U,bm2,rho_b=eigbh(rho)
        print 'Find %s(%s) blocks.'%(bm.nblock,bm2.nblock)

        kpmask=zeros(U.shape[1],dtype='bool')
        spec_cut=sort(spec)[max(0,len(kpmask)-maxN)]
        kpmask[(spec>=spec_cut)&(spec>tol)]=True
        print '%s states kept.'%sum(kpmask)
        trunc_error=sum(spec[~kpmask])
        if direction=='->':
            hgen_l.trunc(U=U,block_marker=bm2,kpmask=kpmask)
        else:
            hgen_r.trunc(U=U,block_marker=bm2,kpmask=kpmask)
        t3=time.time()
        print 'Elapse -> total:%s, eigen:%s'%(t3-t0,t2-t1)
        return e,U,kpmask,trunc_error

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

    def get_mps(self,direction,labels=('s','a'),order=None):
        '''
        Transform <Evolutor> instance to <MPS> instance.

        direction:
            The scan direction.
            '->', right scan.
            '<-', left scan.
        labels:
            (label_site,label_link), The labels for degree of freedom on site and intersite links.
        order:
            The order of indices.
        '''
        assert(direction=='->' or direction=='<-')
        nsite=self.nsite
        if order is None:
            order=NORMAL_ORDER
        if direction=='->':
            hgen=self.query('l',nsite-1)
            opi=self.hchain.query(nsite-1)
            H=hgen.expand(hgen,opi)
            e,v,bm,HH=eigbsh(H,tol=1e-12)
            hgen.trunc(U=v,block_marker=bm,kpmask=ones(1,dtype='bool'))

            ML=[chorder(ai,target_order=order,old_order=[SITE,LLINK,RLINK]) for ai in evolutor.get_AL(dense=True)]
            return MPS(AL=ML,BL=[],S=ones(1),labels=labels)
        else:
            hgen=self.query('r',nsite-1)
            ops=self.hchain.query(0)
            ops=site_image(ops,0,nsite)
            H=hgen.expand(ops)
            e,v,bm,HH=eigbsh(H,tol=1e-12)
            hgen.trunc(U=v,block_marker=bm,kpmask=ones(1,dtype='bool'))
            ML=[chorder(ai,target_order=order,old_order=[SITE,RLINK,LLINK]) for ai in hgen.evolutor.get_AL(dense=True)[::-1]]
            mps=MPS(AL=[],BL=ML,S=ones(1),labels=labels)
            return mps


