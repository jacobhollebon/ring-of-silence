# -----------------------------------------------------------------------------
# Script for the paper:
# "The Ring of Silence in Ambisonics and Binaural Audio Reproduction"
#
# Authors: Jacob Hollebon, Filippo Maria Fazi
#
# Figure 5a and 5b
# Description: Soundfield energy over a circular region reproduced by a circular 
# loudspeaker array performing 2D Ambisonics using measured loudspeaker transfer functions
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

Nfft = 2**13
c = 343 # speed of sound
f = np.linspace(0,24000,int(Nfft/2)+1) # Frequency vector
a = 0.1 # approx head radius
ka = 2*np.pi*f/c *a


dataFolder = repoRoot / "data" # Folder containing the microphone array data

Na = L - N - 1 # order up to which there is no aliasing

#%% Create decoders and perform simulation
data = np.load(dataFolder / "ku100_circular.npz")
phi_dense = data['azimuths'] * np.pi/180    # sampling angles
L_dense = phi_dense.shape[0] # number of sampling point on dense grid

hrir = data['hrir'] 
hrtf_l = np.fft.rfft(hrir[:,0,:],Nfft) # positions x frequency, left ear only


N_dense = L_dense//2
Y_dense = circHarm(az=phi_dense, N=N_dense) # basis fct matrix of CHs sampled at dense positions
h = np.linalg.pinv(Y_dense) @ hrtf_l

Y_N = circHarm(az=phi, N=N)  # basis ftc matrix at virtual speaker and up to order N of CHs

Y_inf = circHarm(az=phi, N=N_dense)  # basis ftc matrix at virtual speaker and up to order N_dense (approx infinity)
x = Y_inf @  h  # resynthesis of HRTF at the virtual loudspeaker array positions (small number)

Ypinv = np.linalg.inv(Y_N.T @ Y_N) @ Y_N.T # least squares inverse
h_tilde = Ypinv @ x   # estimated HRTF Fourier coefficients from set of HRTFs due to loudspeaker array (x) to requested input B-format order (N)

E = np.sum(np.abs(h_tilde)**2,  axis=0)
E_dense = np.sum(np.abs(h)**2,  axis=0)
h_trunc = h[:(2*N+1)]
E_trunc = np.sum(np.abs(h_trunc)**2,  axis=0)

#%% Plot of energy 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(ka, 10*np.log10(abs(E_dense)), label='Reference')
ax.plot(ka, 10*np.log10(abs(E_trunc)),   ls='-.', label = 'Truncated')
ax.plot(ka, 10*np.log10(abs(E)), ls=':', label= 'Reproduced')

ylims = [-15, 15]
ax.set_ylabel(r'SPL$_B$ (dB)')
ax.set_xlabel('ka')
ax.vlines(N, ylims[0], ylims[1], color='0.1', ls='--')
ax.vlines(Na+1, ylims[0], ylims[1], color='0.1', ls='--')
ax.set_xlim(0,44)
ax.set_ylim(ylims[0],ylims[1])
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(True, which='minor', linestyle='--') 
leg = ax.legend(loc=3, framealpha=0.7, facecolor='white', frameon=True)
leg.get_frame().set_linewidth(0.0)
ax.minorticks_on()
fig.tight_layout()
if save:
    pdf = '.pdf'
    fig.savefig(saveFolder / f'figure7a{pdf}')




