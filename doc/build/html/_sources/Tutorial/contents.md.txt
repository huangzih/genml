Tutorial
============

GenML is a Python library designed for generating Mittag-Leffler correlated noise, widely used in modeling complex systems. This notebook demonstrates the basic functionalities of GenML, including generating noise sequences and calculating their autocorrelation functions.<br><br>

## Importation and Parameters
First, get started by having all the necessary tools and libraries imported.

``` python 
import numpy as np
import genml
from matplotlib import pyplot as plt
```
Before diving into the noise generation, it's essential to set up some fundamental parameters that define the properties of the noise we intend to generate, such as the number of sequences, length of each sequence, amplitude coefficient, and others.

```python
# Parameters
N = 500  # Number of sequences
T = 1000  # Length of each sequence
C = 1.0  # Amplitude coefficient
lamda = 1.8  # Mittag-Leffler exponent
tau = 10  # Characteristic memory time
seed = None # Random seed

nc = 4  # Number of CPU cores for parallel processing
tmax = 200  # Max lag for ACF calculation
dt = 1  # Step size between lags
```

## Generating Mittag-Leffler Correlated Noise

In this section, we utilize GenML to generate Mittag-Leffler correlated noise sequences. The `mln` function from the GenML library is called with specified parameters to produce the desired noise sequences. 

```python 
# Generate M-L noise sequences
xi = genml.mln(N, T, C, lamda, tau, seed)
```
### Visualization of Mittag-Leffler Correlated Noise
After generating the Mittag-Leffler noise sequences, we'll visualize them using `matplotlib` to observe the characteristics of each sequence.  

```python 
fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
colors = ['green', 'orange', 'pink']

for i, noise in enumerate(xi[:3, :]):
    axs[i].plot(range(T), noise, color=colors[i])
    axs[i].set_xlim(0, T)
    axs[i].set_ylabel('$\\xi$')
    axs[i].set_xlabel('t')

fig.suptitle(f'M-L Noise with C={C}, λ={lamda}, τ={tau}', y=0.95)

plt.show()

```
![alt f1](../figure/mln_repr.png)

## Calculating Autocorrelation Functions

Autocorrelation functions (ACF) are crucial for understanding the properties of generated noise sequences. Here we calculate both the actual ACF values from the generated sequences and the theoretical ACF values using GenML's `acf` and `acft` functions, respectively.


```python 
# Calculate actual ACF values
acfv = genml.acf(xi, tmax, dt, nc)

# Calculate theoretical ACF values
acftv = genml.acft(tmax, dt, C, lamda, tau)

```

### Visualization of ACF Results

Visualizing the autocorrelation functions helps in comparing the actual and theoretical ACF values, providing insights into the accuracy and effectiveness of the GenML library in simulating Mittag-Leffler correlated noise.

```python 
t = list(range(0, tmax + 1, dt))

plt.plot(t, acfv*tau**lamda/C, 'o', label='acf', color='lightblue', markerfacecolor='none', markersize=12)
plt.plot(t, acftv*tau**lamda/C, label='acft', color='#53A8E1', linewidth=3.6)
plt.legend()

plt.title(f'ACF with N={N}, T={T}, C={C}, λ={lamda}, τ={tau}')
plt.show()

```
![alt f2](../figure/acf_compare.png)

## Validating Seed Effect
The following codes demonstrate the effect of setting a seed value on the generation of random noise sequences. When the `seed` is a specific value, each call to generate noise sequences with the same seed will result in identical sequences. When the `seed` is set to `None`, different runs will produce different noise sequences.

### Same Seeds
```python 
# Generate noise sequences with seed set to 42
xi1 = genml.mln(N, T, C, lamda, tau, seed=42)
xi2 = genml.mln(N, T, C, lamda, tau, seed=42)

# Check if the generated noise sequences are equal
is_equal = np.array_equal(xi1, xi2)
is_equal
```

```python 
t = list(range(0, T))
plt.figure(figsize=(9, 3))
plt.plot(t, xi1[0, :], '-', label='xi1', color='#53A8E1', linewidth=2)
plt.plot(t, xi2[0, :], '--', label='xi2', color='orange', linewidth=2)
plt.legend()

plt.title(f'same seeds')
plt.show()
```
![alt f3](../figure/same_seed.png)

### Different Seeds
```python 
# Generate noise sequences with seed set to different seeds
xi1 = genml.mln(N, T, C, lamda, tau, seed=666)
xi2 = genml.mln(N, T, C, lamda, tau, seed=888)

# Check if the generated noise sequences are equal
is_equal = np.array_equal(xi1, xi2)
is_equal
```

```python 
t = list(range(0, T))
plt.figure(figsize=(9, 3))
plt.plot(t, xi1[0, :], '-', label='xi1', color='#63b2ee', linewidth=2)
plt.plot(t, xi2[0, :], '-', label='xi2', color='#efa666', linewidth=2)
plt.legend()

plt.title(f'different seeds')
plt.show()
```
![alt f4](../figure/diff_seed.png)

### No Seeds
```python
# Generate noise sequences with no seeds
xi1 = genml.mln(N, T, C, lamda, tau, seed=None)
xi2 = genml.mln(N, T, C, lamda, tau, seed=None)

# Check if the generated noise sequences are equal
is_equal = np.array_equal(xi1, xi2)
is_equal
```

```python
t = list(range(0, T))
plt.figure(figsize=(9, 3))
plt.plot(t, xi1[0, :], '-', label='xi1', color='#63b2ee', linewidth=2)
plt.plot(t, xi2[0, :], '-', label='xi2', color='#efa666', linewidth=2)
plt.legend()

plt.title(f'no seeds')
plt.show()
```
![alt f5](../figure/no_seed.png)