import numpy as np
import warnings
import random
from tqdm import tqdm
from .mittag_leffler import ml
import multiprocessing
from multiprocessing import Pool, current_process


class MLN(object):
    """
    Implements the generation of Mittag-Leffler correlated noise.

    This class is dedicated to producing sequences of Mittag-Leffler correlated noise, characterized by its unique
    autocorrelation properties. It is initialized with parameters: T (the sequence length), C (the amplitude coefficient),
    lamda (the Mittag-Leffler exponent), and tau (the characteristic memory time).
    """
    def __init__(self, T, C, lamda, tau, A=None):
        """
        Initializes the MLN class.

        Parameters:
        - T: Total length of the noise trajectory (int).
        - C: Amplitude coefficient (float).
        - lamda: Mittag-Leffler exponent (0 < lamda < 2).
        - tau: Characteristic memory time (0 < tau <= 10000).
        """
        if not isinstance(T, int) or T <= 0:
            raise TypeError("Length of noise must be a positive int.")
        if lamda <= 0 or lamda >= 2:
            raise ValueError("Lamda parameter must be in interval (0, 2).")
        if tau > 10000 or tau <= 0:
            raise ValueError("Tau parameter must be in interval (0, 10000].")
        self.T = T
        self.Topt = T
        self.C = C
        self.lamda = lamda
        self.tau = tau
        self.A = A
        self.acvs = []


    def _mln(self):
        """
        Sample the Mittag-Leffler correlated noise.

        Returns an array of noise sequence.
        """
        Z = np.random.normal(0.0, 1.0, 2 * self.T)
        return self.syn_mln(Z)


    def cal_acft(self, lag):
        """
        Calculate the theoretical autocorrelation function for Mittag-Leffler correlated noise.

        Parameters:
        - lag (int or array): The time lag(s).

        Returns:
        - The calculated autocorrelation function value(s) at the specified lag(s).
        """
        lag = -(abs(lag) / self.tau) ** self.lamda
        return self.C * ml(lag, alpha=self.lamda) / (self.tau ** self.lamda)


    def gen_A(self):
        """
        Computes the eigenvalues of a circulant matrix with optimal length, ensuring non-negativity.

        This function identifies the optimal sequence length and computes the eigenvalues of the associated circulant matrix
        derived from the autocorrelation function. The process ensures the non-negativity condition. 
        
        The 'ls' list serves as a sorted waiting list of potential sequence lengths.

        Returns:
        - An array of non-negative eigenvalues corresponding to the optimal length circulant matrix.
        """
        ls0 = [int(25 * i) for i in range(4, 40)]
        ls = list(range(1, 10)) + [int(2.5 * i) for i in range(4, 40)] + ls0 + list(np.array(ls0) * 10)\
            + list(np.array(ls0) * 100) + list(np.array(ls0) * 1000) + list(np.array(ls0) * 10000) + list(np.array(ls0) * 100000)\
            + list(np.array(ls0) * 1000000) + list(np.array(ls0) * 10000000) + list(np.array(ls0) * 100000000) + list(np.array(ls0) * 1000000000)
        self.acvs = self.cal_acft(np.array(range(self.T))).tolist()
        first_row = self.acvs + [self.cal_acft(self.T)] + self.acvs[1:][::-1]
        A0 = np.fft.fft(first_row).real
        if np.any(A0 < 0):
            warnings.warn("Not meeting the nonnegativity condition, trajectory will be extracted from longer trajectories.")
            ls_ = [x for x in ls if x > self.Topt]
            for i in ls_:
                T1 = self.Topt
                self.Topt = i
                add_acvs = self.cal_acft(np.array(range(T1, self.Topt))).tolist()
                self.acvs += add_acvs
                first_row = self.acvs + [self.cal_acft(self.Topt)] + self.acvs[1:][::-1]
                A1 = np.fft.fft(first_row).real
                if not np.any(A1 < 0):
                    print(f"Optimal length found: T_opt = {self.Topt}")
                    return A1
            return None
        else:
            return A0


    def syn_mln(self, Z):
        """
        Synthesizes Mittag-Leffler correlated noise leveraging Fourier techniques.

        This function transforms a sequence of normal random variables into Mittag-Leffler correlated noise, based on Fourier techniques.

        Parameters:
        - Z (array): A sequence of normal random variables.

        Returns:
        - An array representing the synthesized Mittag-Leffler correlated noise.
        """
        A = self.gen_A()
        Z = np.random.normal(0.0, 1.0, 2 * self.T)
        Y = np.zeros(2 * self.T, dtype=complex)
        for i in range(2 * self.T):
            A_sqrt = np.sqrt(A[i] * self.T)
            if i == 0:
                Y[i] = A_sqrt * np.sqrt(2) * Z[i]
            elif i < self.T:
                Y[i] = A_sqrt * (Z[2 * i - 1] + 1j * Z[2 * i])
            elif i == self.T:
                Y[i] = A_sqrt * np.sqrt(2) * Z[2 * self.T - 1]
            else:
                Y[i] = Y[2 * self.T - i].conjugate()
        # Fourier transform of the combined noise sequence
        X = np.fft.ifft(Y)
        return X[:self.T].real



def cal_mln(args):
    """
    Instantiate the MLN class to generate a single noise sequence.
    """
    T, C, lamda, tau, A = args
    return MLN(T, C, lamda, tau, A)._mln()


def cal_acf(args):
    """
    Computes the actual autocorrelation function of Mittag-Leffler correlated noise at a specific lag.
    """
    xi, T, i = args
    if i == 0:
        corr = np.sum(xi[:, :] * xi[:, :], axis=1) / T
    else:
        corr = np.sum(xi[:, :-i] * xi[:, i:], axis=1) / (T - i)
    return corr.mean()




def mln(N, T, C, lamda, tau, nc=1):
    """
    Generates N Mittag-Leffler correlated noise sequences of length T in parallel.
    Adjusts noise length to match the target length if required.

    Parameters:
    - N: Number of trajectories to generate (int).
    - T: Total length of the noise trajectory (int).
    - C: Amplitude coefficient (float).
    - lamda: Mittag-Leffler exponent (0 < lamda < 2).
    - tau: Characteristic memory time (0 < tau <= 10000).
    - nc: Number of parallel cores (int).

    Returns:
    A 2D array of size (N, T) containing the generated Mittag-Leffler correlated noise trajectories.
    """
    if not isinstance(N, int) or N <= 0:
        raise TypeError("Number of trajectories must be a positive int.")
    if current_process().name == 'MainProcess':
        pbar = tqdm(total=N, desc='Generating M-L noise', position=0)
    xi = np.zeros((N, T))
    A = MLN(T, C, lamda, tau).gen_A()
    Topt = A.shape[0] // 2
    args_ls = [(Topt, C, lamda, tau, A) for _ in range(N)]
    npc = multiprocessing.cpu_count() // 2
    nc = min(N, npc) if nc > N else min(nc, npc)
    with Pool(processes=nc) as pool:
        for i, tmp in enumerate(pool.imap(cal_mln, args_ls)):
            rd = random.randint(0, Topt - T)
            xi[i, :] = tmp[rd:rd + T]
            if pbar: pbar.update(1)
    if pbar: pbar.close()
    return xi


def acf(xi, tmax, dt, nc=1):
    """
    Parallel computation of the actual autocorrelation function for specified lags for Mittag-Leffler correlated noise sequences.

    Parameters:
    - xi: 2D numpy array containing noise sequences generated by `genml.mln`.
    - tmax: Maximum lag to compute the autocorrelation function for (int).
    - dt: Step size between lags (int).
    - nc: Number of parallel cores (int).

    Returns:
    A 1D numpy array containing the acf values.
    """
    if not isinstance(xi, np.ndarray):
        raise TypeError("xi must be a numpy ndarray")
    if not isinstance(tmax, int) or not isinstance(dt, int) or not isinstance(nc, int):
        raise TypeError("tmax, dt, and nc must be integers")
    if np.any(xi.shape) <= 0:
        raise ValueError("The number and length of the noise must be greater than 0.")
    T = xi.shape[1]
    if tmax > T - 1:
        raise ValueError("The max lag must be lower than T.")
    if not (1 <= dt < tmax):
        raise ValueError("The sampling interval must be within the range [1, tmax).")
    npc = multiprocessing.cpu_count() // 2
    nc = min(tmax, npc) if nc > tmax else min(nc, npc)
    ls = list(range(0, tmax + 1, dt))
    args_ls = [(xi, T, i) for i in ls]
    with Pool(processes=nc) as pool:
        acfv = list(tqdm(pool.imap(cal_acf, args_ls), total=len(args_ls), desc='Calculating ACF', position=0))
    return np.array(acfv)


def acft(tmax, dt, C, lamda, tau):
    """
    Calculate the theoretical autocorrelation function for Mittag-Leffler correlated noise.

    Parameters:
    - tmax: Maximum lag to compute the acft for (int).
    - dt: Step size between lags (int).
    - C: Amplitude coefficient (float).
    - lamda: Mittag-Leffler exponent (0 < lamda < 2).
    - tau: Characteristic memory time (0 < tau <= 10000).

    Returns:
    A 1D numpy array containing the theoretical acf.
    """
    if not (1 <= dt < tmax):
        raise ValueError("The sampling interval must be within the range [1, tmax).")
    acftv = MLN(tmax, C, lamda, tau).cal_acft(np.array(range(0, tmax + 1, dt)))
    return acftv
