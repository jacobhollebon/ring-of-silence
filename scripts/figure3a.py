# -----------------------------------------------------------------------------
# Script for the paper:
# "The Ring of Silence in Ambisonics and Binaural Audio Reproduction"
#
# Authors: Jacob Hollebon, Filippo Maria Fazi
#
# Figure 3a
# Description: Average energy reproduced by a circular loudspeaker array performing 2D Ambisonics
#
# This script supports the analyses and results presented in the above paper.
# Please cite the paper if you use or adapt this code for academic purposes.
#
# License: Creative Commons Attribution 4.0 International (CC BY 4.0)
# https://creativecommons.org/licenses/by/4.0/
# -----------------------------------------------------------------------------

#%%% Toolbox imports
# Core toolboxes
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from hos.toolboxes.circular import circHarm
# Plot formatting
import scienceplots
plt.style.use(['science','ieee'])
import matplotlib as mpl; mpl.rcParams['figure.dpi'] = 200;  mpl.rcParams['savefig.dpi'] = 300

#%% Figure handling

# Find path to save figures to location "repo/figures"
repoRoot = Path(__file__).parent.parent.resolve()
saveFolder = repoRoot / "figures"
saveFolder.mkdir(parents=True, exist_ok=True) # Make the folder if it does not exist

save = True # Bool to trigger saving the figures

#%% User setup
CHdef = 'real' # 'real' or 'exp': type of circular harmonics

N  = 2 # order of the b-format input signal
N_ = 6  # order of the loudspeaker array

L  = 2*N_+1 # number of speakers in array
phi = (np.arange(L)+0.5) *2*np.pi/L # Loudspeaker positions, uniform sampling, add 0.5 to avoid speaker at 0 degrees

#%% Build input and reference b-format signa;s

kr = np.arange(0,20,.01)
kr = kr[1:] # removing 0 Hz value
c = 344 # speed of sound
Na = L - N - 1 # order up to which there is no aliasing

# Reference b format signal of a plane wave from az = 0 degs
N_dense = int(kr[-1]*4) # approximates infty
b = np.squeeze(circHarm(az=0, N=N_dense, kind=CHdef)) # b format signals up to N_dense

# Input b format signal, truncated version of the reference
b_T = b[:2*N+1] # Truncated B-format signal

#%% Decoders and reproduced energy

Y_N = circHarm(az=phi, N=N, kind=CHdef) # Matrix of CHs sampled by loudspeakers to the truncated order
Y_inf = circHarm(az=phi, N=N_dense, kind=CHdef) # Matrix of CHs sampled by loudspeakers to infty
D = np.linalg.pinv(Y_N).T # Pseudoinverse decoder
A = Y_inf.T @ D # aliasing matrix
b_tilde = A @ b_T

E = np.zeros(kr.shape) # Energy, reference input field
E_T = np.zeros(kr.shape) # Energy, truncated input field
E_R = np.zeros(kr.shape) # Energy, reproduced by loudspeaker array and truncated input field
Jn = np.zeros((kr.shape[0],2*N_dense+1)) # Radial functions
Jn_max = np.zeros(2*N_dense+1)
N_dense_nidxs = np.zeros(2*N_dense+1) # List of order numbers for each Bformat channel
N_dense_ACNidxs = np.arange(2*N_dense+1)  # List of ACN numbering for each Bformat channel
for i in range(1, N_dense+1):
    N_dense_nidxs[(2*(i-1)+1) : (2*i+1)] = int(i)

# Perform simulation per order and summing
for i in range(2*N_dense+1):
    n = N_dense_nidxs[i]
    Rn =  2*np.pi * special.jv(n, kr) # radial fct
    Jn[:,i] = Rn.copy() 
    Jn_max[i] = kr[np.argmax(Rn)]
    E += np.abs(Rn * b[i])**2
    E_T += np.abs(Rn * b[i]*(n<=N))**2
    E_R += np.abs(Rn * b_tilde[i])**2

normFactor = 1/(2*np.pi) # 1.4pi for 3D, 1/2pi for 2D
E = E * normFactor
E_T = E_T * normFactor
E_R = E_R * normFactor

#%% Energy plots across kr

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(kr,10*np.log10(E), label='Reference')
ax1.plot(kr,10*np.log10(E_T),  ls='-.', label = 'Truncated')
ax1.plot(kr,10*np.log10(E_R), ls=':', label= 'Reproduced')

ylims = [-15, 5]
ax1.vlines(N, ylims[0], ylims[1], color='0.1', ls='--')
ax1.vlines(Na+1, ylims[0], ylims[1], color='0.1', ls='--')

ax1.set_xlabel(r'kr');
ax1.set_ylabel(r'SPL (dB)')
ax1.set_ylim(ylims[0],ylims[1])
ax1.set_yticks([-15,-10,-5,0,5])
ax1.set_xlim(0,20)
ax1.grid(True, which='major', linestyle='-')
ax1.grid(True, which='minor', linestyle='--') 
ax1.minorticks_on()
leg = ax1.legend(loc=3, framealpha=0.7, facecolor='white', frameon=True)
leg.get_frame().set_linewidth(0.0)
fig1.tight_layout()
fig1.show()


if save:
    pdf = '.pdf'
    fig1.savefig(saveFolder / f'figure3a{pdf}')
    