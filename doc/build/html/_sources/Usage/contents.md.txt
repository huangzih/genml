Usage
========

This section introduces the basic usage of GenML. GenML provides several interfaces for efficiently generating M-L noise sequences in batches and computing actual and theoretical autocorrelation functions. The following are detailed explanations:

### Parameters

- `N` [type `int`]: Number of noise sequences to generate.
- `T` [type `int`]: Length of the noise sequences to generate.
- `C` [type `float`]: Amplitude coefficient.
- `lamda` [type `float`]: Mittag-Leffler exponent, range (0, 2).
- `tau` [type `int`]: Characteristic memory time, range (0, 10000).
- `seed` [default=`None`]: Random seed for generating noise.
   - Set to `None` for different noise sequences each time; set to a specific value for identical noise sequences each time.
- `xi` [type `ndarray`]: Generated noise sequences, shape (N, T).
- `tmax` [type `int`]: Maximum value for computing autocorrelation functions, range (0, T).
- `dt` [type `int`]: Interval for computing autocorrelation functions, range (0, tmax).
- `nc` [default=1]: Number of cores to use for parallel computation of autocorrelation functions.
---
> **Note**: When `lamda` approaches 2 and `tau` is large, generating noise sequences may take a long time and consume a significant amount of memory. It is recommended to use caution in such cases.


### Generating Mittag-Leffler Correlated Noise

To generate M-L noise, use the `mln` interface with the following method: `genml.mln(N, T, C, lamda, tau, seed)`.<br><br>
Example usage:

```python
import genml

# Parameters
N = 10  # Number of sequences
T = 500  # Length of each sequence
C = 1.0  # Amplitude coefficient
lamda = 0.5  # Mittag-Leffler exponent
tau = 10  # Characteristic memory time
seed = 42 # Random seed

# Generate M-L noise sequences
xi = genml.mln(N, T, C, lamda, tau, seed)
```

### Calculating Autocorrelation Function

To calculate the autocorrelation function (ACF) values of the generated noise sequences, you can use the `acf` function for actual ACF values and the `acft` function for theoretical ACF values. Usage: `genml.acf(xi, tmax, dt, nc)` and `genml.acft(tmax, dt, C, lamda, tau)`.<br><br>
Example usage:
```python
tmax = 100  # Max lag for ACF calculation
dt = 1  # Step size between lags
nc = 4  # Number of CPU cores for parallel processing

# Calculate actual ACF values
acfv = genml.acf(xi, tmax, dt, nc)

# Calculate theoretical ACF values
acftv = genml.acft(tmax, dt, C, lamda, tau)
```
