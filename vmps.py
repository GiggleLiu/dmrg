'''
Variational Matrix Product State.
'''

from numpy import *
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix,coo_matrix
from matplotlib.pyplot import *
import time,pdb

from rglib.mps import NORMAL_ORDER,BHKContraction,contract,tensor

__all__=['VMPSEngine']

ZERO_REF=1e-15

class VMPSEngine(object):
    '''
    Variational MPS Engine.

    Parameters
    -------------------
    H:
        The hamiltonian operator.
    ket:
        The <MPS>.
    contractor_L/contractor_R:
        The contractor from left and right.
    '''
    def __init__(self,H,k0):
        self.H=H
        self.ket=k0
        self.ket<<self.ket.l  #right normalize the ket
        self.l=0

        nsite=k0.nsite
        bra=self.bra
        self.contractor_L=BHKContraction(bra,self.H,self.ket,l=0)
        self.contractor_R=BHKContraction(bra,self.H,self.ket,l=nsite)
        #first, left sweep to construct R.
        self.contractor_R<<self.nsite
        #self.contractor_L>>self.nsite
        #ion()
        #self.contractor_R.show()

    @property
    def bra(self):
        '''Get the bra.'''
        return self.ket.tobra(labels=[self.H.labels[0],self.ket.labels[1]+"'"],sharedata=True)

    @property
    def nsite(self):
        '''Number of sites.'''
        return self.ket.nsite

    def update_move(self,M,direction,tol=1e-8):
        '''
        Update the ket and move towards specific direction.

        direction:

            * '->', move rightward.
            * '<-', move leftward.
        tol:
            The tolerence for compression.
        '''
        #check validity of datas
        assert(direction=='->' or direction=='<-')
        nsite=self.nsite
        l=self.l
        if (l>=nsite and direction=='->') or (l<0 and direction=='<-'):
            raise ValueError('Can not perform %s move for l = %s with %s sites in total.'%(direction,l,nsite))

        ###!!!!'Ket or Bra, check!'
        if direction=='->':
            self.ket.update_right(M,tol=tol)
        else:
            self.ket.update_left(M,tol=tol)
        bra=self.bra
        self.contractor_L.set_ket(self.ket)
        self.contractor_L.set_bra(bra)
        self.contractor_R.set_ket(self.ket)
        self.contractor_R.set_bra(bra)
        if direction=='->':
            self.contractor_L>>1
            self.l+=1
        else:
            self.contractor_R<<1
            self.l-=1

    def fix_boundary(self,turning_to=''):
        '''Fix the boundary and make ready for a turnning.'''
        if turning_to=='':
            turning_to='->' if self.nsite==-1 else '<-'
        assert(turning_to=='->' or turning_to=='<-')
        if turning_to=='->':
            assert(self.l==-1)
            self.l+=1
            self.contractor_L=BHKContraction(self.bra,self.H,self.ket,l=0)
        else:
            nsite=self.nsite
            assert(self.l==nsite)
            self.l-=1
            self.contractor_R=BHKContraction(self.bra,self.H,self.ket,l=nsite)

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
