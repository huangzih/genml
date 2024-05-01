Introduction
================

As a ubiquitous stochastic process in complex systems, noise is integral to the evolution of such systems. The intrinsic physical properties of noise are essential for understanding the mechanisms underlying system behaviors. The basic theoretical description of noise in various studies is the white noise, which exhibits a time-independent feature and a uniform power spectrum. However, unexpected behavior patterns in complex systems are often induced by non-white noise with temporal correlations. Numerous studies have supported this perspective, among which are representative examples: in ultracold and collisionless atomic gases, non-white noise modulates the light scattering growth rate based on its correlation time; in the survival dynamics of biological populations, non-white environmental noise affects the dependency of population extinction times; and in thermophoretic effects, non-white noise directs the motion of Brownian particles in fluids along the temperature gradient. These findings highlight the pivotal role of non-white noise in revealing the internal workings of complex systems, and emphasize the necessity of considering noise effects when elucidating the evolution dynamics.

In particular, **Mittag-Leffler correlated noise** (referred to as **M-L noise**) is distinguished for its capability to emulate a broad spectrum of correlation behaviors through parameter adjustments, including typical exponential and power-law correlations. This flexibility makes M-L noise a versatile tool for modeling a wide range of phenomena in complex systems. For example,  Laas et al. utilize M-L noise to offer a theoretical exploration of the resonance behaviors observed in Brownian particles within oscillatory viscoelastic shear flows. Cairano et al. underscore M-L noise's role in unraveling the crossover from subdiffusive to Brownian motion and the relaxation dynamics within membrane proteins. Umamaheswari et al. demonstrate the application of M-L noise in financial mathematics by leveraging it to explain the existence and stability of solutions in nonlinear stochastic fractional dynamical systems. That is, the importance of M-L noise in dissecting complex system dynamics is significant, necessitating advancements in its generation for deeper insights.

However, despite significant efforts by researchers to develop algorithms for simulating various types of non-white noise, direct algorithms and software for generating M-L noise remain notably absent. Such an absence significantly impedes its application in crucial simulation methods such as Langevin dynamics and molecular dynamics, as well as in data-driven approaches like machine learning. This limitation confines the utility of M-L noise primarily to theoretical modeling, thereby restricting in-depth exploration of its effects on complex systems. Consequently, there is a great demand for the development of tools capable of accurately simulating M-L noise, which could unlock new insights into the dynamic behaviors of complex systems across various scientific fields.

To address this critical issue, we introduce **GenML**, a Python library designed to effectively generate M-L noise in this document. This software marks a significant advancement, enabling researchers to directly simulate M-L noise and apply it across a wide range of fields. GenML not only fills the existing gap in the available simulation tools for M-L noise but also paves the way for new research opportunities in understanding and modeling the dynamics of complex systems.

