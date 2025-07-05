# -----------------------------------------------------------------------------
# Script for the paper:
# "The Ring of Silence in Ambisonics and Binaural Audio Reproduction"
#
# Authors: Jacob Hollebon, Filippo Maria Fazi
#
# Figure 6, 7b and 7c
# Description: Energy analysis of HRTF SH coefficients and reproduced HRTFs using 3D Ambisonics
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
from hos.toolboxes.spherical import sphHarm, nmACN
# Plot formatting
import scienceplots
plt.style.use(['science','ieee'])
import matplotlib as mpl; mpl.rcParams['figure.dpi'] = 200;  mpl.rcParams['savefig.dpi'] = 300

#%% Figure handling

# Find path to save figures to location "repo/figures"
repoRoot = Path(__file__).parent.parent.resolve()
saveFolder = repoRoot / "figures"
saveFolder.mkdir(parents=True, exist_ok=True) # Make the folder if it does not exist

save = True  # Bool to trigger saving the figures

#%% User setup
SHdef = 'realn3d' # Type of spherical harmonics: 'realn3d', 'real'==SN3D, or 'complex'

N  = 2 # order of the b-format input signal
N_ = 6  # order of the loudspeaker array 
t = 12  # t order of the t design
Na = t - N  # order up to which there is no aliasing
N_dense = 40 # order up to which reference is calculated, approximates infty

# Loudspeaker positions
dataFolder = repoRoot / "data" 
speakerPositionFiles = ['spherical_O6_84pt_t12.npy', 'spherical_O6_equalarea.npy']

Nfft = 2**13
c = 343 # speed of sound
f = np.linspace(0,24000,int(Nfft/2)+1) # Frequency vector
a = 0.1 # approx head radius
c = 344 # speed of sound
ka = 2*np.pi*f/c *a


#%% Build reference HRTF SH coefficients and
data = np.load(dataFolder / "ku100_2702.npz")
phi_dense = data['grid']
hrir = data['hrir']
hrtf = np.fft.rfft(hrir, axis=-1, n=Nfft) 
Y_dense = sphHarm(pos=phi_dense, N=N_dense, kind=SHdef)
h = np.linalg.pinv(Y_dense) @ hrtf[:,0,:] # SH HRTF coefficents up to reference order, approximates infty, just keep left ear

#%%

for idx, currSpeakerPositionFile in enumerate(speakerPositionFiles):
        
    # Load and format the sampling positions
    loadPath = dataFolder / currSpeakerPositionFile
    samplingPositions = np.load(loadPath) 
    samplingPositions = samplingPositions.T 
    samplingPositions[:,0] = samplingPositions[:,0] % 360 # Map azimuth to between 0-360
    samplingPositions = samplingPositions[:,:2] # remove radial
    samplingPositions = np.deg2rad(samplingPositions)
    samplingPositions = np.concatenate((samplingPositions, np.ones((samplingPositions.shape[0], 1))), axis=1) # add radius of 1 for compatability with HOS toolbox
    L  = samplingPositions.shape[0] # number of speakers in array
    
    # SHs sampled at reproduction sampling points
    Y_N = sphHarm(pos=samplingPositions, N=N, kind=SHdef, plot=True)
    
    # SHs sampled at reproduction sampling points up to reference infty order
    Y_inf = sphHarm(pos=samplingPositions, N=N_dense, kind=SHdef)
    x = Y_inf @  h  # resynthesis of HRTF at the reproduction sampling positions (small number)
    
    Ypinv = np.linalg.pinv(Y_N) # pseudoinverse decoder
    h_tilde = Ypinv @ x  # estimated HRTF Fourier coefficients from set of HRTFs due to loudspeaker array (x) to requested input B-format order (N)
        
    E = np.sum(np.abs(h_tilde)**2,  axis=0)
    E_dense = np.sum(np.abs(h)**2,  axis=0)
    h_trunc = h[:(2*N+1)]
    E_trunc = np.sum(np.abs(h_trunc)**2,  axis=0)
    
    
    #%% Plot energy of HRTF SH coefficients
    
    plotOrders = np.arange(0,5,1)
    plotACN = plotOrders**2 + plotOrders + 0 # use m=0 only
    norm = np.sum(np.abs(h),axis=0)
    ylims = [-60, 0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['C0','C1','C2','C3','m']
    for i, order in enumerate(plotOrders):
        color = colors[i]
        idxstart = order**2 + order - order
        idxstop = order**2 + order + order
        
        sumEnergy = np.sum(np.abs(h[idxstart:idxstop+1,:]), axis=0)
        
        lab = r'$n=$'+str(order)
        
        ax.semilogx(ka, 20*np.log10(abs(sumEnergy/norm)), label=lab, color=color)
        
    leg = ax.legend(loc=3, framealpha=0.7, facecolor='white', frameon=True)
    leg.get_frame().set_linewidth(0.0)
    ax.set_ylabel(r'$h$ (dB)')
    ax.set_xlabel('ka')
    ax.set_xlim(0,44)
    ax.set_ylim(ylims[0],ylims[1])
    ax.grid(visible=True, which='major', linestyle='-')
    ax.grid(True, which='minor', linestyle='--') 
    ax.minorticks_on()
    fig.tight_layout()
    if save:
        pdf = '.pdf'
        fig.savefig(saveFolder / f'figure6{pdf}')
        
    #%% Plot energy of average HRTF
    
    ylims = [-30, 20]
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    
    ax2.plot(ka, 10*np.log10(abs(E_dense)), label='Reference')
    ax2.plot(ka, 10*np.log10(abs(E_trunc)),   ls='-.', label = 'Truncated')
    ax2.plot(ka, 10*np.log10(abs(E)), ls=':', label= 'Reproduced')
    
    ax2.set_ylabel(r'SPL$_B$ (dB)')
    ax2.set_xlabel('ka')
    ax2.vlines(N, ylims[0], ylims[1], color='0.1', ls='--')
    ax2.vlines(Na+1, ylims[0], ylims[1], color='0.1', ls='--')
    ax2.set_xlim(0,44)
    ax2.set_ylim(ylims[0],ylims[1])
    ax2.grid(visible=True, which='major', linestyle='-')
    ax2.grid(True, which='minor', linestyle='--') 
    leg = ax2.legend(loc=3, framealpha=0.7, facecolor='white', frameon=True)
    leg.get_frame().set_linewidth(0.0)
    ax2.minorticks_on()
    fig2.tight_layout()
    if save:
        label = chr(ord('b') + idx)  # 'a', 'b', 'c', ...
        pdf = '.pdf'
        fig2.savefig(saveFolder / f'figure7{label}{pdf}')
    
    
    


