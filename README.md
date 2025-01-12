# GenML: A Python Library to Generate the Mittag-Leffler Correlated Noise

> Xiang Qu, Hui Zhao, Wenjie Cai, Gongyi Wang, and [Zihan Huang](https://grzy.hnu.edu.cn/site/index/huangzihan)

*School of Physics and Electronics, Hunan University, Changsha 410082, China*

E-mail: huangzih@hnu.edu.cn

GenML is a Python library designed for generating Mittag-Leffler correlated noise, which is crucial for modeling a wide range of phenomena in complex systems. This document provides a brief overview of how to install and use GenML to generate M-L noise and compute its autocorrelation functions.

This work is supported by the National Natural Science Foundation of China (Grant No. 12104147) and the Fundamental Research Funds for the Central Universities.

##### Documentation website: [https://genml.readthedocs.io](https://genml.readthedocs.io)

## Installation

To install GenML, simply run the following command in your Python environment:

```bash
pip install -U genml
```

## Basic Usage

The core functionalities of GenML include generating sequences of Mittag-Leffler correlated noise and calculating their autocorrelation functions. Here's how you can get started:

### Generating Mittag-Leffler Correlated Noise

To generate sequences of Mittag-Leffler correlated noise , use the `mln` function with the desired parameters:

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

To calculate the autocorrelation function (ACF) values of the generated noise sequences, you can use the `acf` function for actual ACF values and the `acft` function for theoretical ACF values:

```python
tmax = 100  # Max lag for ACF calculation
dt = 1  # Step size between lags
nc = 4  # Number of CPU cores for parallel processing

# Calculate actual ACF values
acfv = genml.acf(xi, tmax, dt, nc)

# Calculate theoretical ACF values
acftv = genml.acft(tmax, dt, C, lamda, tau)
```

### Examples

The repository includes detailed examples illustrating the generation of Mittag-Leffler correlated noise and the calculation of its autocorrelation function. These examples demonstrate the library's capability to replicate theoretical noise properties.
