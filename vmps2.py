'''
Variational Matrix Product State.
'''

from numpy import *
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix,coo_matrix
from matplotlib.pyplot import *
import time,pdb

from rglib.mps import NORMAL_ORDER,BHKContraction,contract,tensor

__all__=['VMPS2Engine']

ZERO_REF=1e-15

def split_opstring(ops,l):
    '''
    Split an operator string into two parts at bond-l.

    Parameters:
        :ops: <OpString>, the operator string.
        :l: int, the bond.

    Return:
        len-2 tuple, two opstrings.
    '''
    return OpString(filter(lambda ou:ou.siteindex<l,ops.opunits)),OpString(filter(lambda ou:ou.siteindex>=l,ops.opunits))

def op_repr(op,ket,code):
    '''
    Operator representation at bond-l.

    Parameters:
        :op: <OpString>/<OpUnit>, the target operator(compact form required).
        :ket: <MPS>, the state.
        :code: len-2 str, e.g. '<5' for left part up to 5-th bond, '>4' for right part up to 4-th bond.

    Return:
        <Tensor>, rank-2 matrix.
    '''
    rightmove,l=code[0]=='<',int(code[1])
    if isinstance(op,OpUnit):
        op=OpString([op])
    sites=op.siteindex
    nsite=ket.nsite
    site_axis=ket.site_axis
    rlink_axis=ket.rlink_axis
    llink_axis=ket.llink_axis
    lmin,lmax=min(sites),max(sites)
    if rightmove:
        lmax=l-1
        iterator=xrange(lmin,lmax+1)
    else:
        lmin=l
        iterator=xrange(lmax,lmin-1,-1)

    res=None
    for i in iterator:
        items=[]
        M=ket.get(i,attach_S='B')
        MH=bra.get(i,attach_S='B')
        if i==lmin and rightmove:
            MH.labels[llink_axis]=M.labels[llink_axis]
        else:
            MH.labels[llink_axis]=M.labels[llink_axis]+'\''
        if i==lmax and not rightmove:
            MH.labels[rlink_axis]=M.labels[rlink_axis]
        else:
            MH.labels[rlink_axis]=M.labels[rlink_axis]+'\''
        if i in sites:
            opunit=op.query(i)[0]
            MH.labels[site_axis]=M.labels[site_axis]+'\''
            O=Tensor(opunit.get_data(),labels=[MH.labels[site_axis],M.labels[site_axis]])
            items.append(O)
        else:
            MH.labels[site_axis]=M.labels[site_axis]
        items.extend([M,MH])
        for item in items:
            res=contract(res,item) if res is not None else item
    return res

def opc_repr_update(H0,mps,op_added,code):
    '''
    Evolve the Hamiltonian from H0 to new one by adding one site to the right.
    
    Parameters:
        :H0: <Tensor>, matrix representation for operator in the last step.
        :mps: <MPS>, the state.
        :op_added: list of <OpString>, the new operator containing current site.
        :code: len-2 str, e.g. '<5' for left part up to 5-th bond, '>4' for right part up to 4-th bond.

    Return:
        <Tensor>, the new representation for operator.
    '''
    rightmove,l=code[0]=='<',int(code[1])
    assert(mps.l<=l)  #make sure the mps is left canonical upto l.
    hndim=mps.hndim
    if rightmove:
        #new site wave function.
        Ai=mps.get(l-1,attach_S='B')
        #the original hamiltonian.
        Ai.chlabel(mps.llink_axis,H0.labels[1])
        Bi=Ai.conj()
        Bi.chlabel(mps.llink_axis,H0.labels[0])
    else:
        #new site wave function.
        Ai=mps.get(l,attach_S='A')
        #the original hamiltonian.
        Ai.chlabel(mps.rlink_axis,H0.labels[1])
        Bi=Ai.conj()
        Bi.chlabel(mps.rlink_axis,H0.labels[0])
    H0=Bi*H0*Ai
    #the new term.
    #first, query the relevant operators
    for op in op_added:
        H0=H0+op_repr(op,mps,code=code)
    return H0

def opc_repr(mps,opc,code):
    '''
    Representation for operator collection. It's faster than the bruteforce way(through update).

    Parameters:
        :mps: <MPS>, the state with specific canonicality(left/right canonical if right/left move).
        :opc: <OpCollection>, the operators.
        :code: len-2 str, e.g. '<5' for left part up to 5-th bond, '>4' for right part up to 4-th bond.

    Return:
        <Tensor>, matrix representation of the operator.
    '''
    rightmove,l=code[0]=='<',int(code[1])
    nsite=mps.nsite
    if rightmove:
        H=Tensor(zeros([1,1]),labels=['a_0\'','a_0'])
        for li in xrange(l):
            nop=filter(lambda op:all(op.siteindex<=li),opc.query(li))  #site NL and NL+1
            H=opc_repr_update(H,mps,op_added=nop,code='<%s'%(li+1))
    else:
        H=Tensor(zeros([1,1]),labels=['a_N\'','a_N'])
        for li in xrange(nsite-1,l-1,-1):
            nop=filter(lambda op:all(op.siteindex>=li),opc.query(li))  #site NL and NL+1
            H=opc_repr_update(H,mps,op_added=nop,code='>%s'%(li))
    return H

class VMPS2Engine(object):
    '''
    Variational MPS Engine with 2-site update.

    Attributes:
        hchain: <OpCollection>, the hamiltonian operator.
        ket: <MPS>, the state.
    '''
    def __init__(self,hchain):
        self.hchain=hchain
        self.l=0
        self.LPART={}   #memory of left and right hailtonian history.
        self.RPART={}

    def set_initial_mps(self,k0=None):
        '''
        Set up the initial mps.
        '''
        #first, prepair the state.
        if k0 is None:
            k0=random_product_state(nsite=self.hchain.nsite,hndim=self.hchain.hndim)
        self.ket=k0
        self.ket<<self.ket.l  #right normalize the ket
        
        #initialize left hamiltonian dicts, because we start at 0-th bond, we need only the 0-th item.
        H=Tensor(zeros([1,1]),labels=['a_0\'','a_0'])
        self.LPART[0]=H
        #for li in xrange(l):
        #    nop=filter(lambda op:all(op.siteindex<=li),opc.query(li))  #site NL and NL+1
        #    H=opc_repr_update(H,mps,op_added=nop,code='<%s'%(li+1))
        #    self.LPART[li+1]=H

        #initialize right hamiltonian dicts.
        H=Tensor(zeros([1,1]),labels=['a_N\'','a_N'])
        self.RPART[0]=H
        for li in xrange(nsite-1,l-1,-1):
            nop=filter(lambda op:all(op.siteindex>=li),opc.query(li))  #site NL and NL+1
            H=opc_repr_update(H,mps,op_added=nop,code='>%s'%(li))
            self.RPART[nsite-li]=H

    @property
    def nsite(self):
        '''Number of sites.'''
        return self.ket.nsite

    def takeet_hmatrix(self,which,H,l):
                '''
                        Set the hamiltonian for specific length.
                        
                                Parameters:
                                                :which: str,
                                                
                                                                * `l` -> the left part.
                                                                                * `r` -> the right part.
                                                                                            :H: <Tensor>, the hamiltonian.
                                                                                                        :l: int, the length of block.
                                                                                                                '''
                                                                                                                        if which=='l':
                                                                                                                                        self.LPART[l]=H
                                                                                                                                                else:
                                                                                                                                                                self.RPART[l]=H
                                                                                                                                                                
                                                                                                                                                                self,rightmove=True):
        '''
        Take a step towards right.
        '''

    def get_optimal_ket(self,L,W,R):
        '''
        Optimize the state at current position.

        L/R:
            The left contraction and right contraction.
        W:
            The MPO at this site.

        *return*:
            tuple of (EG per site, optimal M-matrix of current position)
        '''
        t=contract(L,W,R)
        #reshape into square sparse matrix.
        if len(t.labels)!=6:
            raise Exception('Wrong number of vertices, check your labels!')
        llink_axis,rlink_axis=self.ket.llink_axis,self.ket.rlink_axis
        t=transpose(t,axes=(0,2,4,1,3,5))  #this is the normal order
        n=prod(t.shape[:3])
        mat=t.reshape([n,n])
        mat[abs(mat)<ZERO_REF]=0
        mat=csr_matrix(mat)
        Emin,vec=eigsh(mat,which='SA',k=1,tol=1e-12,maxiter=5000)
        M=tensor.Tensor(vec.reshape(t.shape[:3]),labels=t.labels[3:])
        M=tensor.chorder(M,old_order=NORMAL_ORDER,target_order=self.ket.order)
        return Emin/self.nsite,M

    def run(self,maxiter=10,tol=0):
        '''
        Run this application.

        maxiter:
            The maximum sweep iteration, one iteration is contains one left sweep, and one right sweep through the chain.
        tol:
            The tolerence.
        '''
        elist=[]
        nsite=self.nsite
        L=self.contractor_L.get_memory(self.l)
        R=self.contractor_R.get_memory(self.l+1)
        W=self.H.get(self.l)
        for iiter in xrange(maxiter):
            print '########### STARTING NEW ITERATION %s ################'%iiter
            for direction,iterator in zip(['->','<-'],[xrange(nsite),xrange(nsite-1,-1,-1)]):
                for i in iterator:
                    print 'Running iter = %s, direction = %s, n = %s'%(iiter,direction,i)
                    t0=time.time()
                    L=self.contractor_L.get_memory(self.l)
                    R=self.contractor_R.get_memory(self.l+1)
                    W=self.H.get(self.l)
                    Emin,M=self.get_optimal_ket(L,W,R)
                    elist.append(Emin)
                    self.update_move(M,direction=direction)
                    t1=time.time()
                    diff=Inf if len(elist)<=1 else elist[-1]-elist[-2]
                    print 'Get Emin/site = %.12f, tol = %s, Elapse -> %s'%(Emin,diff,t1-t0)
                self.fix_boundary(turning_to='<-' if direction=='->' else '->')
            if iiter==0:
                Emin_last=elist[0]
            diff=Emin_last-Emin
            print 'ITERATION SUMMARY: Emin/site = %s, tol = %s, Elapse -> %s'%(Emin,diff,t1-t0)
            if abs(diff)<tol:
                print 'Converged, Breaking!'
                return Emin
            else:
                Emin_last=Emin
