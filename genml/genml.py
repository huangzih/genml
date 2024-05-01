import sys
import time
import warnings
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from .mittag_leffler import ml
from multiprocess import Pool as Pool1
from multiprocessing import Pool as Pool2


class MLN(object):
    """
    Implement the generation of Mittag-Leffler correlated noise.

    This class is dedicated to producing sequences of Mittag-Leffler correlated noise, characterized by its unique
    autocorrelation properties. It is initialized with parameters: N (number of noise sequences), T (the sequence length), 
    C (the amplitude coefficient), lamda (the Mittag-Leffler exponent), tau (the characteristic memory time) and seed (random seed).
    """
    
    A = None
    params = None
   
    def __init__(self, N, T, C, lamda, tau, seed=None):
        """
        Initialize the MLN class.

        Parameters:
        - N: Number of noise sequences to generate (int).
        - T: Total length of single noise sequence (int).
        - C: Amplitude coefficient (float).
        - lamda: Mittag-Leffler exponent (0 < lamda < 2).
        - tau: Characteristic memory time (0 < tau <= 10000).
        - seed: Random seed (default: None).
        """
        if not isinstance(T, int) or T <= 0:
            raise TypeError("Length of noise must be a positive int.")
        if lamda <= 0 or lamda >= 2:
            raise ValueError("Lamda parameter must be in interval (0, 2).")
        if tau > 10000 or tau <= 0:
            raise ValueError("Tau parameter must be in interval (0, 10000].")
        self.N = N
        self.T = T
        self.Topt = T
        self.C = C
        self.lamda = lamda
        self.tau = tau
        self.seed = seed
        self.acvs = []
        

    def _mln(self):
        """
        Sample the Mittag-Leffler correlated noise.

        Return an array of noise sequences.
        """
        np.random.seed(self.seed)
        Z = np.random.normal(0.0, 1.0, (self.N, 2 * self.T))
        return self.syn_mln(Z)


    def cal_acft(self, lag):
        """
        Calculate the theoretical autocorrelation function for Mittag-Leffler correlated noise.

        Parameters:
        - lag (int or array): The time lag(s).

        Return:
        - The calculated autocorrelation function value(s) at the specified lag(s).
        """
        lag = -(abs(lag) / self.tau) ** self.lamda
        return self.C * ml(lag, alpha=self.lamda) / (self.tau ** self.lamda)


    def update_A(self):
        '''
        Update the matrix of non-negative eigenvalues based on current parameters or availability.

        Check if MLN.A is unavailable or parameters have changed. If conditions are met, trigger the 
        generation of a new MLN.A using 'gen_A'.
        '''
        if MLN.A is None or MLN.params != (self.T, self.C, self.lamda, self.tau):
            MLN.A = self.gen_A()
            MLN.params = (self.T, self.C, self.lamda, self.tau)            

    
    def gen_A(self):
        """
        Compute the eigenvalues of a circulant matrix with optimal length, ensuring non-negativity.

        This function identifies the optimal sequence length and computes the eigenvalues of the associated circulant matrix
        derived from the autocorrelation function. The process ensures the non-negativity condition. 
        
        The 'ls' list serves as a sorted waiting list of potential sequence lengths.

        Return:
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
        Synthesize Mittag-Leffler correlated noise leveraging Fourier techniques.

        This function transforms a sequence of normal random variables into Mittag-Leffler correlated noise, based on Fourier techniques.

        Parameters:
        - Z (array): Sequences of normal random variables.

        Return:
        - An array representing the synthesized Mittag-Leffler correlated noise.
        """
        A = MLN.A
        Y = np.zeros((self.N, 2 * self.T), dtype=complex)
        A_sqrt = np.sqrt(A * self.T)
        Y[:, 0] = A_sqrt[0] * np.sqrt(2) * Z[:, 0]
        Y[:, 1:self.T] = A_sqrt[1:self.T] * (Z[:, 1:2*self.T-1:2] + 1j * Z[:, 2:2*self.T:2])
        Y[:, self.T] = A_sqrt[self.T] * np.sqrt(2) * Z[:, 2 * self.T - 1]
        Y[:, self.T+1:] = Y[:, self.T-1:0:-1].conjugate()
        X = np.fft.ifft(Y)
        return X[:, :self.T].real


def cal_mln(args):
    """
    Instantiate the MLN class to generate noise sequences.
    """
    N, T, C, lamda, tau, seed = args
    return MLN(N, T, C, lamda, tau, seed)._mln()


def cal_acf(args):
    """
    Compute the actual autocorrelation function of Mittag-Leffler correlated noise at a specific lag.
    """
    xi, T, i = args
    if i == 0:
        corr = np.sum(xi[:, :] * xi[:, :], axis=1) / T
    else:
        corr = np.sum(xi[:, :-i] * xi[:, i:], axis=1) / (T - i)
    return corr.mean()


def mln(N, T, C, lamda, tau, seed=None):
    """
    Generate N Mittag-Leffler correlated noise sequences of length T.
    Adjusts noise length to match the target length if required.

    Parameters:
    - N: Number of noise sequences to generate (int).
    - T: Total length of single noise sequence (int).
    - C: Amplitude coefficient (float).
    - lamda: Mittag-Leffler exponent (0 < lamda < 2).
    - tau: Characteristic memory time (0 < tau <= 10000).
    - seed: Random seed (default: None).

    Return:
    A 2D array of size (N, T) containing the generated Mittag-Leffler correlated noise sequences.
    """
    if not isinstance(N, int) or N <= 0:
        raise TypeError("Number of sequences must be a positive int.")
    start_time = time.time()
    xi = np.zeros((N, T))
    MLN(N, T, C, lamda, tau).update_A()
    Topt = MLN.A.shape[0] // 2
    np.random.seed(seed)
    rd = np.random.randint(0, Topt - T + 1, N)
    idx = np.arange(T) + rd[:, np.newaxis]
    args = N, Topt, C, lamda, tau, seed
    xi = cal_mln(args)[np.arange(N)[:, np.newaxis], idx]
    end_time = time.time()
    execution_time = end_time - start_time
    print("M-L Noise Generation Time: ", execution_time, "s")
    return xi


def acf(xi, tmax, dt, nc=1):
    """
    Parallel computation of the actual autocorrelation function for specified lags for Mittag-Leffler correlated noise sequences.

    Parameters:
    - xi: 2D numpy array containing noise sequences generated by `genml.mln`.
    - tmax: Maximum lag to compute the autocorrelation function for (int).
    - dt: Step size between lags (int).
    - nc: Number of parallel cores (int).

    Return:
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
    npc = mp.cpu_count()
    nc = min(tmax, npc) if nc > tmax else min(nc, npc)
    ls = list(range(0, tmax + 1, dt))
    args_ls = [(xi, T, i) for i in ls]
    if sys.platform.startswith('darwin'):
        print("macOS detected, employing the multiprocess library")
        with Pool1(processes=nc) as pool:
            acfv = list(tqdm(pool.imap(cal_acf, args_ls), total=len(args_ls), desc='Calculating ACF', position=0))
    else:
        with Pool2(processes=nc) as pool:
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

    Return:
    A 1D numpy array containing the theoretical acf.
    """
    if not (1 <= dt < tmax):
        raise ValueError("The sampling interval must be within the range [1, tmax).")
    acftv = MLN(1, tmax, C, lamda, tau).cal_acft(np.array(range(0, tmax + 1, dt)))
    return acftv
