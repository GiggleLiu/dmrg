from numpy import *
import copy

from pymps.mpo import *
from pymps.mpolib import *
from tba.hgen import SpinSpaceConfig
from tba.lattice import Square_Lattice

class HeisenbergModel():
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
    def __init__(self,J,Jz,h,nsite,nspin=3):
        self.spaceconfig=SpinSpaceConfig([1,nspin])
        scfg=self.spaceconfig
        I=OpUnitI(hndim=scfg.hndim)
        Sx=opunit_Sx(spaceconfig=scfg)
        Sy=opunit_Sy(spaceconfig=scfg)
        Sz=opunit_Sz(spaceconfig=scfg)
        Sp=opunit_Sp(spaceconfig=scfg)
        Sm=opunit_Sm(spaceconfig=scfg)
        wi=zeros((5,5),dtype='O')
        #wi[:,0],wi[4,1:]=(I,Sp,Sm,Sz,-h*Sz),(J/2.*Sm,J/2.*Sp,Jz*Sz,I)
        wi[:,0],wi[4,1:]=(I,Sx,Sy,Sz,-h*Sz),(J*Sx,J*Sy,Jz*Sz,I)
        WL=[copy.deepcopy(wi) for i in xrange(nsite)]
        WL[0]=WL[0][4:]
        WL[-1]=WL[-1][:,:1]
        self.H=WL2MPO(WL)
        mpc=WL2OPC(WL)
        self.H_serial=mpc

class HeisenbergModel2D():
    '''
    2D Heisenberg model application for vMPS.

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
    def __init__(self,J,Jz,h,N1,N2,nspin=3):
        self.spaceconfig=SpinSpaceConfig([1,nspin])
        scfg=self.spaceconfig
        I=OpUnitI(hndim=scfg.hndim)
        Sx=opunit_Sx(spaceconfig=scfg)
        Sy=opunit_Sy(spaceconfig=scfg)
        Sz=opunit_Sz(spaceconfig=scfg)
        Sp=opunit_Sp(spaceconfig=scfg)
        Sm=opunit_Sm(spaceconfig=scfg)
        opc=OpCollection()
        lt=Square_Lattice((N1,N2))
        lt.set_periodic([True,True])
        lt.initbonds(5)
        bonds=[b for b in lt.getbonds(1) if b.atom2>b.atom1]
        for b in bonds:
            atom1,atom2=b.atom1,b.atom2
            opc+=J*(Sx.as_site(atom1)*Sx.as_site(atom2)+Sy.as_site(atom1)*Sy.as_site(atom2))
            opc+=Jz*(Sz.as_site(atom1)*Sz.as_site(atom2))
        for i in xrange(lt.nsite):
            opc=opc+h*Sz.as_site(i)
        mpo=opc.toMPO(nsite=lt.nsite,method='direct')
        self.H=mpo
