import numpy as np
import matplotlib.pyplot as plt

import qutip as qt

def d2op(N, delta):
    """Second derivative operator.

    This function returns a discretized second derivative operator using the 
    central difference method.

    Args:
        N (int): The system dimention is given by `2*N + 1`
        delta (float): The step size of the system.

    Returns:
        numpy.array: A NxN array representing the discretized second derivative
        in matrix form.

    References:
        [1]: https://en.wikipedia.org/wiki/Finite_difference
    """
    return (np.diag(np.ones(2*N), 1) - 2 * np.diag(np.ones(2*N + 1)) + np.diag(np.ones(2*N), -1)) / (delta**2)


class Transmon(object):
    """A transmon qubit.

    This class models a transmon qubit by creating and diagonalizing the
    hamiltonian in the phase basis.
    """
    def __init__(self, Ec, Ej, num_phis=200, units='Hz', **kwargs):
        """Initializes the Transmon class.

        Args:
            Ec (float): The charging energy (Ec) of the transmon in units of
                frequency. This will be in Hz if `units` is `Hz`. Otherwise, it
                is assumed to be in units of angular frequency.
            Ej (float): The Josephson energy (Ej) of the transmon. The unit
                conventions are the same as for `Ec`.
            num_phis (int): This gives the number of basis elements in the phase
                basis between 0 and pi. The hamiltonian is diagonalized between
                -pi and pi, so the total number of basis elements will be
                2*num_phis + 1
        
        References:
            [1]: See https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.042319
                for the original transmon paper.
            [2]: See https://arxiv.org/abs/1904.06560 for a more modern introduction
                to the Transmon.
        """
        super().__init__()

        if units == 'Hz':
            scaling = 2*np.pi
        else:
            scaling = 1

        self.Ec = Ec * scaling
        self.Ej = Ej * scaling
        self.H = self.hamiltonian(num_phis)

        self.ens, self.evs = self.H.eigenstates()
        self.ens -= self.ens[0]
        self.omega = self.ens[1] - self.ens[0]
        self.alpha = self.ens[2] - 2*self.ens[1]
        
    def hamiltonian(self, num_phis):
        """Constructs the Transmon hamiltonian.

        The transmon hamiltonian is given by
            $$H = -4*E_C \partial^2_\varphi - E_J \cos(\varphi)$$
        in the phase basis. This implicitly assumes there is no offset charge.

        Args:
            num_phis: The number of basis elements between 0 and pi.
                The total dimension of the Hamiltonian will be `2*num_phis + 1`.

        Returns:
            qutip.Qobj: The transmon Hamiltonian in the phase basis.
        """
        Ec = self.Ec
        Ej = self.Ej
        self.phis = np.linspace(-np.pi, np.pi, 2 * num_phis + 1)
        Hc = -4 * Ec * d2op(num_phis, self.phis[1] - self.phis[0])
        Hj = -Ej * np.diag(np.cos(self.phis))

        return qt.Qobj(Hc + Hj)

    def eigenbasis(self, N):
        """Returns the transmon Hamiltonian in the eigenbasis, truncated to N levels.

        Args:
            N (int): The number of levels to truncate the Hamiltonian to.
        
        Returns:
            qutip.Qobj: The transmon Hamiltonian in the eigenbasis.
        """
        return qt.Qobj(np.diag(self.ens[:N]))

    def lowering_op(self, N):
        """Returns a modified "lowering" operator for the transmon.

        This "lowering" operator is defined such that when multiplied with its
        hermitian conjugate gives the transmon hamiltonian scaled such that the
        1 state energy is exactly 1.

        This should not be used to couple a transmon to a microwave drive field,
        but rather when describing the coupling between two transmons.
        
        Args:
            N (int): The numbe of levels to truncate the operator to.

        Returns:
            qutip.Qobj: The modified lowering operator.
        """
        return qt.Qobj(np.diag(np.sqrt(self.ens[1:N] / self.ens[1]), 1))

    def raising_op(self, N):
        """Returns a modified "raising" operator for the transmon.

        This "raising" operator is defined such that when multiplied with its
        hermitian conjugate gives the transmon hamiltonian scaled such that the
        1 state energy is exactly 1.

        This should not be used to couple a transmon to a microwave drive field,
        but rather when describing the coupling between two transmons.
        
        Args:
            N (int): The numbe of levels to truncate the operator to.

        Returns:
            qutip.Qobj: The modified lowering operator.
        """
        return qt.Qobj(np.diag(np.sqrt(self.ens[1:N] / self.ens[1]), -1))

    @staticmethod
    def compute_transmon_parameters(omega, alpha):
        """Computes Ej and Ec given the frequency and anharmonicity of the qubit.

        This function uses the fourth order taylor expansion of $\omega$ and 
        $\alpha$ to solve $E_C$ and $E_J$. The taylor expansions are in terms
        of the small parameter $\eta = \sqrt{\frac{2E_C}{E_J}}$, as given in
        [1]. 
        
        The units of omega and alpha, in principle, do not matter as long as
        they are consistent. However, working in units of GHz tends to give
        better results.

        Args:
            omega (float): The frequency (01) of the transmon. See above on units.
            alpha (float): The anharmonicity of the transmon, defined as f12 - f01. 
                This should be a negative number. See above on units.

        Returns:
            tuple: A tuple (Ec, Ej) with the same units as omega and alpha.

        References:
            [1]: See https://arxiv.org/abs/1706.06566.
        """
        p_a = np.poly1d([46899/(2**15), 4635/(2**12), 81/(2**7), 9/(2**4), 1])
        p_w = np.poly1d([-5319/(2**15), -19/(2**7), -21/(2**7), -1/4, -1, 4])

        eta = np.poly1d([1, 0])

        p_t = eta*omega*p_a + alpha*p_w

        roots = p_t.r[-1]

        eta_0 = np.real(roots[np.isreal(roots)][0])

        if not np.isclose(p_t(eta_0), 0):
            print('Error: Unable to find a solution.')

        Ec = -alpha/p_a(eta_0)

        Ej = 2*Ec/(eta_0**2)

        return Ec, Ej

class DuffingOscillator(object):
    """docstring for DuffingOscillator"""

    def __init__(self, omega, alpha, N, units='Hz', **kwargs):
        super().__init__()

        if units == 'Hz':
            scaling = 2*np.pi
        else:
            scaling = 1

        self.omega = omega * scaling
        self.alpha = alpha * scaling
        self.N = N
        self.H = self.hamiltonian()

        self.ens, self.evs = self.H.eigenstates()
        self.ens -= self.ens[0]
    
    def hamiltonian(self):
        N = self.N
        H = self.omega * create(N) * destroy(N)
        H += self.alpha/2 * create(N) * create(N) * destroy(N) * destroy(N)

        return H

    def eigenbasis(self, N):
        return Qobj(np.diag(self.ens[:N]))

    def lowering_op(self, N):
        return destroy(N)

    def raising_op(self, N):
        return create(N)


