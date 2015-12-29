'''
Tests
'''

from .models import *

def test_model():
    '''
    Test for Heisenberg model.
    '''
    model=HeisenbergModel(J=1.,Jz=0.8,h=0.1,nsite=6)
    ion()
    model.H.show()
    show()
    pdb.set_trace()
    mpc=model.H.serialize()
    mpc.compactify()
    mpo2=mpc.toMPO()
    mpc2=mpo2.serialize()
    print mpo2
    cla()
    mpc2.show()
    pdb.set_trace()


