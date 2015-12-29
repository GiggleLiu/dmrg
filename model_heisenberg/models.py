'''
Heisenberg Model.
'''

from numpy import *
from matplotlib.pyplot import *
import pdb,time,copy
from mps.core.mpo import MPO,opunit_I
from mps.core.mpolib import opunit_Sz,opunit_Sp,opunit_Sm,opunit_Sx,opunit_Sy
from mps.vmpsapp import VMPSApp
from tba.hgen import SpinSpaceConfig

class HeisenbergModel(VMPSApp):
    '''
    Heisenberg model application for vMPS.

    The Hamiltonian is: sum_i J/2*(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + Jz*S_i^zS_{i+1}^z -h*S_i^z

    Construct
    -----------------
    HeisenbergModel(J,Jz,h)

    Attributes
    ------------------
    J/Jz:
        Exchange interaction at xy direction and z direction.
    h:
        The strength of weiss field.
    *see VMPSApp for more attributes.*
    '''
    def __init__(self,J,Jz,h,nsite):
        self.spaceconfig=SpinSpaceConfig([2,1])
        I=opunit_I(hndim=2)
        Sz=opunit_Sz(spaceconfig=[2,1])
        Sp=opunit_Sp(spaceconfig=[2,1])
        Sm=opunit_Sm(spaceconfig=[2,1])
        wi=zeros((5,5),dtype='O')
        wi[:,0],wi[4,1:]=(I,Sp,Sm,Sz,-h*Sz),(J/2.*Sm,J/2.*Sp,Jz*Sz,I)
        WL=[copy.deepcopy(wi) for i in xrange(nsite)]
        WL[0]=WL[0][4:]
        WL[-1]=WL[-1][:,:1]
        self.H=MPO(WL)
        mpc=self.H.serialize()
        mpc.compactify()
        self.H_serial=mpc

class HeisenbergModel2(VMPSApp):
    '''
    Heisenberg model application for vMPS.

    The Hamiltonian is: sum_i J/2*(S_i^+S_{i+1}^- + S_i^-S_{i+1}^+) + Jz*S_i^zS_{i+1}^z -h*S_i^z

    Construct
    -----------------
    HeisenbergModel(J,Jz,h)

    Attributes
    ------------------
    J/Jz/J2/Jz2:
        Exchange interaction at xy direction and z direction, and the same parameter for second nearest hopping term.
    h:
        The strength of weiss field.
    *see VMPSApp for more attributes.*
    '''
    def __init__(self,J,Jz,h,nsite,J2=0,J2z=None):
        self.spaceconfig=SpinSpaceConfig([2,1])
        I=opunit_I(hndim=2)
        Sx=opunit_Sx(spaceconfig=[2,1])
        Sy=opunit_Sy(spaceconfig=[2,1])
        Sz=opunit_Sz(spaceconfig=[2,1])

        #with second nearest hopping terms.
        if J2==0:
            self.H_serial2=None
            return
        elif J2z==None:
            J2z=J2
        SL=[]
        for i in xrange(nsite):
            Sxi,Syi,Szi=copy.copy(Sx),copy.copy(Sy),copy.copy(Sz)
            Sxi.siteindex=i
            Syi.siteindex=i
            Szi.siteindex=i
            SL.append(array([Sxi,Syi,sqrt(J2z/J2)*Szi]))
        ops=[]
        for i in xrange(nsite-1):
            ops.append(J*SL[i].dot(SL[i+1]))
            if i<nsite-2 and J2z!=0:
                ops.append(J2*SL[i].dot(SL[i+2]))

        mpc=sum(ops)
        self.H_serial=mpc
        ion()
        mpc.show_advanced()
        pdb.set_trace()
