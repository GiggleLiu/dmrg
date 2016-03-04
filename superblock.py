'''
Super Block class.
'''

from numpy import *
import scipy.sparse as sps
import copy,time,pdb,warnings

from rglib.hexpand import Z4scfg,kron
from rglib.mps import OpString,OpUnit

__all__=['site_image','SuperBlock']

def site_image(ops,NL,NR):
    '''
    Perform imaging transformation for operator sites.
    
    Parameters:
        :ops: list of <OpString>/<OpUnit>, the operator(s) for operation.
        :NL/NR: integer, the number of sites in left/right block.

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
    
    Construct:
        SuperBlock(hl,hr)

    Attributes:
        :hl/hr: <RGHGen>, Hamiltonian Generator for left and right blocks.
        :order: 'A.B.'/'A..B', the space ordering.
        :nsite: integer, number of total sites(readonly).
        :hndim: integer, the dimension of a single site(readonly).
    '''
    def __init__(self,hl,hr,order='A.B.'):
        self.hl=hl
        self.hr=hr
        assert(order=='A..B' or order=='A.B.')
        self.order=order

    @property
    def nsite(self):
        '''Total number of sites'''
        return self.hl.N+self.hr.N+(2 if self.hl.truncated else 0)

    @property
    def hndim(self):
        '''The hamiltonian dimension of a single site.'''
        return self.hl.hndim

    def site_image(self,ops):
        '''
        Perform imaging transformation for operator sites.
        
        Parameters:
            :ops: list of <OpString>/<OpUnit>, the operator(s) for operation.

        Return:
            list of <OpString>/<OpUnit>, the operators after imaginary operation.
        '''
        NL,NR=self.hl.N+(1 if self.hl.truncated else 0),self.hr.N+(1 if self.hr.truncated else 0)
        return site_image(ops,NL,NR)

    def get_op_onlink(self,ouA,ouB):
        '''
        Get the operator on the link.
        
        Parameters:
            :ouA/ouB: <OpUnit>, the opunit on left/right link site.

        Return:
            matrix, the hamiltonian term.
        '''
        NL,NR=self.hl.N+(1 if self.hl.truncated else 0),self.hr.N+(1 if self.hr.truncated else 0)
        scfg=self.hl.spaceconfig
        ndiml0=self.hl.evolutor.check_link(NL-1)
        ndimr0=self.hr.evolutor.check_link(NR-1)
        #assert(ouA.siteindex==NL-1 and ouB.siteindex==NR-1)
        if ouA.fermionic:
            sgn=Z4scfg(scfg)
            if self.order=='A.B.':
                sgnr=self.hr.zstring.get(NR-1)
                assert(sgnr is not None)
                mA=kron(sps.identity(ndiml0),sps.csr_matrix(ouA.get_data()).dot(sgn))
                mB=kron(sgnr,sps.csr_matrix(ouB.get_data()))
            else:
                mA=kron(sps.identity(ndiml0),sps.csr_matrix(ouA.get_data()).dot(sgn))
                mB=kron(sps.csr_matrix(ouB.get_data()),sps.identity(ndimr0))
        else:
            if self.order=='A.B.':
                mA=kron(sps.identity(ndiml0),sps.csr_matrix(ouA.get_data()))
                mB=kron(sps.identity(ndimr0),sps.csr_matrix(ouB.get_data()))
            else:
                mA=kron(sps.identity(ndiml0),sps.csr_matrix(ouA.get_data()))
                mB=kron(sps.csr_matrix(ouB.get_data()),sps.identity(ndimr0))
        op=kron(mA,mB)
        return op

    def get_op(self,opstring):
        '''
        Get specific operator.
        '''
        if self.order=='A.B.':
            return self._get_op_AdBd(opstring)
        else:
            return self._get_op_AddB(opstring)

    def _get_op_AdBd(self,opstring):
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
        NL,NR=self.hl.N+(1 if self.hl.truncated else 0),self.hr.N+(1 if self.hr.truncated else 0)
        if any(array(siteindices)>=self.nsite):
            raise ValueError('Site index out of range.')
        if not (len(siteindices)==len(unique(siteindices)) and all(diff(siteindices)>=0)):
            raise ValueError('Compact opstring is required!')
        sgn=Z4scfg(self.hl.spaceconfig)
        I1=sps.identity(hndim)
        op_ll=filter(lambda ou:ou.siteindex<NL-1,opstring.opunits)
        op_ls=filter(lambda ou:ou.siteindex==NL-1,opstring.opunits)
        op_rs=self.site_image(filter(lambda ou:ou.siteindex==NL,opstring.opunits))
        op_rr=self.site_image(filter(lambda ou:ou.siteindex>NL,opstring.opunits))
        nll,nls,nrs,nrr=len(op_ll),len(op_ls),len(op_rs),len(op_rr)

        if nll+nls==0 or nrs+nrr==0:
            raise NotImplementedError('Only inter-block terms are allowed,\
                    this is not implemented by perpose to avoid missing hamiltonian terms in DMRG iteration.')
        elif  hasattr(self.hr,'zstring'):
            #handle the fermionic link.
            if nll!=0 or nrr!=0:
                raise NotImplementedError('Only nearest neighbor term is allowed for fermionic links!')
            return self.get_op_onlink(op_ls[0],op_rs[0])

        datas=[]
        for hgen,opn,op1,NN in [(self.hl,op_ll,op_ls,NL),(self.hr,op_rr,op_rs,NR)]:
            ndim0=hgen.evolutor.check_link(NN-1)
            #get the data in opn-block
            if len(opn)>0:
                opstr=prod(opn)
                if isinstance(opstr,OpString):
                    opstr.compactify()
                data_n=kron(hgen.get_op(opstr,target_len=NN-1),I1)
            else:
                data_n=None

            #get the data in op1-block
            if len(op1)>0:
                ou=op1[0]
                if ou.fermionic:
                    sgnr=hgen.zstring.get(NN-1).diagonal()
                    data_1=sps.kron(sgnr,ou.get_data())
                else:
                    data_1=sps.kron(identity(ndim0),ou.get_data())
            else:
                data_1=None

            if data_n is None and data_1 is not None:
                data=data_1
            elif data_n is not None and data_1 is None:
                data=data_n
            else:
                data=data_n.dot(data_1)
            datas.append(data)
        return kron(datas[0],datas[1])


    def _get_op_AddB(self,opstring):
        '''
        Get the hamiltonian from a opstring instance.

        Parameters:
            :opstring: <OpString>, the operator string, it must be compactified!

        Return:
            matrix, the hamiltonian term.
        '''
        hndim=self.hndim
        raise Exception('Not Tested!')
        siteindices=list(opstring.siteindex)
        nsite=self.nsite
        NL,NR=self.hl.N+1,self.hr.N+1
        if any(array(siteindices)>=self.nsite):
            raise ValueError('Site index out of range.')
        if not (len(siteindices)==len(unique(siteindices)) and all(diff(siteindices)>=0)):
            raise ValueError('Compact opstring is required!')
        sgn=Z4scfg(self.hl.spaceconfig)
        I1=sps.identity(hndim)
        op_ll=filter(lambda ou:ou.siteindex<NL-1,opstring.opunits)
        op_ls=filter(lambda ou:ou.siteindex<NL-1,opstring.opunits)
        op_rs=filter(lambda ou:ou.siteindex==NL,opstring.opunits)
        op_rr=filter(lambda ou:ou.siteindex>NL,opstring.opunits)
        nll,nls,nrl,nrr=len(op_ll),len(op_ls),len(op_rs),len(op_rr)
        #handle the fermionic link.
        if nll+nls>0 and nrs+nrr>0 and hasattr(self.hr,'zstring'):
            if nll!=0 or nrr!=0:
                raise NotImplementedError('Only nearest neighbor term is allowed for fermionic links!')
            return self.get_op_onlink(op_ls[0],op_rs[0])
        if nll>0:
            opstr=prod(op_ll).compactify()
            opl=self.hl.get_op(opstr,target_len=NL-1)
            if len(ou_pre)!=0:
                opstr=prod(ou_pre)
                if isinstance(opstr,OpString):
                    opstr.compactify()
                opl=self.hl.get_op(opstr,target_len=NL-1)
                opl=kron(opl,sps.identity(hndim))
                if opstr.fermionic:
                    fermionic=~fermionic
            else:
                opl=sps.identity(self.hr.ndim*hndim)
            if len(ou_cur)!=0:
                ou=ou_cur[0]
                if hasattr(self.hr,'zstring') and ou.fermionic:
                    I0=self.hr.zstring[NL-1]
                else:
                    I0=identity(self.hr.ndim)
                ou0=ou.get_data()
                opl=opl.dot(kron(I0,ou0))
                if ou.fermionic:
                    fermionic=~fermionic
        else:
            opl=sps.identity(self.hl.ndim*hndim)
        if len(oprl)>0:
            opstr=prod(oprl)
            if isinstance(opstr,OpString):
                opstr.compactify()
                opunits=opstr.opunits
            else:
                opunits=[opstr]
            if len(opunits)==0 and ou0.fermionic:
                raise NotImplementedError('This operator is overall fermionic, which is not implemented!')
            if opunits[-1].siteindex!=NR-1:
                if hasattr(self.hr,'zstring'):  #fermioic generator
                    opr0,fermi0=self.hr.get_op(opstr,target_len=NR-1,get_fermionic_sign=True)
                    if fermi0:
                        raise NotImplementedError('This operator is overall fermionic, which is not implemented!')
                else:
                    opr0=self.hr.get_op(opstr,target_len=NR-1)
                opr=kron(sps.identity(hndim),opr0)
            else:
                ou0=opunits.pop(-1)
                if len(opunits)==0:
                    opr0,fermi0=identity(self.hr.evolutor.check_link(NR-1)),False
                elif hasattr(self.hr,'zstring'):  #fermioic generator
                    opr0,fermi0=self.hr.get_op(OpString(opunits),target_len=NR-1,get_fermionic_sign=True)
                else:
                    opr0,fermi0=self.hr.get_op(OpString(opunits),target_len=NR-1),False
                if fermi0^ou0.fermionic:
                    raise NotImplementedError('This operator is overall fermionic, which is not implemented!')
                elif fermi0:
                    op0=ou0.get_data().dot(sgn)
                else:
                    op0=ou0.get_data()
                opr=kron(op0,opr0)
        else:
            opr=sps.identity(self.hr.ndim*hndim)
        return kron(opl,opr)


