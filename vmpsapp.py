'''
Application for vMPS.
'''

__all__=['VMPSApp']

class VMPSApp(object):
    '''
    The App class for vMPS

    Attributes:
        :Ham: The hamiltonian in the form of <MPO>.
        :k0: The starting <MPS> ket.
    '''
    def __init__(self,Ham,k0):
        self.Ham=Ham
        self.k0=k0

    @property
    def nsite(self):
        '''The number of sites.'''
        return len(self.Ham)
