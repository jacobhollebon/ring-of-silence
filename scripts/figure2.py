# -----------------------------------------------------------------------------
# Script for the paper:
# "The Ring of Silence in Ambisonics and Binaural Audio Reproduction"
#
# Authors: Jacob Hollebon, Filippo Maria Fazi
#
# Figure 2
# Description: Aliasing analysis of a spherical loudspeaker array performing 3D Ambisonics
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
from hos.toolboxes.spherical import nmACN, sphHarm, orthogonalityMatrix
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

#%% User parameters
N = 2 # Order of reproduction
Ndash = 12 #  Order up to which aliasing is evaluated

SHdef = 'realn3d' # Type of spherical harmonics: 'realn3d', 'real'==SN3D, or 'complex'

# Loudspeaker positions
dataFolder = repoRoot / "data" 
speakerPositionFiles = ['spherical_O6_84pt_t12.npy', 'spherical_O6_equalarea.npy']

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
   
    # Variables containing all SH indicies
    n, m, ACN = nmACN(N)
    SHindxs = np.stack((m, n, ACN)).T
    
    # Create the SH matrix, aliasing and orthogonality checks
    Ynm1 = sphHarm(pos=samplingPositions, N=N, kind=SHdef) # up to order of reproduction
    O = orthogonalityMatrix(Ynm1)
    
    #%% Aliasing matrix
    
    Ynm2 = sphHarm(pos=samplingPositions, N=Ndash, kind=SHdef) # up to order of aliasing evaluation
    Ynm1pinv = np.linalg.pinv(Ynm1) # Pseudoinverse decoder
    E = np.abs(  Ynm1pinv @ Ynm2 ) 
    E[E < 0.000001] = 0.000001 # clipping to avoid log10(0)
    
    #%% Plot of the aliasing matrix
    
    # Hlines for each order up to N
    maxN_hlines = int(np.sqrt(Ynm1.shape[1])-1)
    n_hlines = np.arange(0,maxN_hlines+1)
    ACN_hlines = (n_hlines**2)-0.5
    numCoeffsYnm2 = Ynm2.shape[1]
    # Vlines for each order up to Ndash
    maxN_vlines = int(np.sqrt(Ynm2.shape[1])-1)
    n_vlines = np.arange(0,maxN_vlines+1)
    ACN_vlines = (n_vlines**2)-0.5
    numCoeffsYnm1 = Ynm1.shape[1]
    
    
    
    fig = plt.figure(figsize=(6.6, 1.35)) 
    ax1 = fig.add_subplot(1, 1, 1)
    
    im = ax1.imshow(20*np.log10(abs(E)), vmax=0, vmin=-60, aspect="auto", cmap='Greys_r')
    ax1.hlines(ACN_hlines, -0.5, numCoeffsYnm2, 'w')
    ax1.vlines(ACN_vlines, -0.5, numCoeffsYnm1, 'w')
    ax1.set_ylim(-0.5, numCoeffsYnm1-0.5)
    ax1.set_xlim(-0.5, numCoeffsYnm2-0.5)
    
    xticks = np.arange(0,Ndash+1)**2 + (2*np.arange(0,Ndash+1))
    ax1.set_xticks(xticks)
    yticks = np.arange(0,N+1)**2 + (2*np.arange(0,N+1))
    ax1.set_yticks(yticks)
    
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()
    
    orderlabelsaliased =[]
    for i in range(Ndash+1):
        j=-i
        orderlabelsaliased.append(i)
    orderticksrep = []
    orderlabelsrep =[]
    for i in range(N+1):
        j=-i
        orderlabelsrep.append(i)
        
    orderticksaliased = (ACN_vlines+0.5)/(Ndash**2 + Ndash + Ndash)
    orderticksaliased[6:] -= (0.35)/(Ndash**2 + Ndash + Ndash)
    orderticksaliased[10:] -= (0.5)/(Ndash**2 + Ndash + Ndash)
    ax2.set_xticks(orderticksaliased[1:])
    ax2.set_xticklabels(orderlabelsaliased[1:])
    ax2.set_xlabel("Order")
    
    orderticksrep = ACN_hlines/(N**2 + N + N)
    ax3.set_yticks(orderticksrep[1:])
    ax3.set_yticklabels(orderlabelsrep[1:])
    ax3.set_ylabel("Order")
    
    ax1.set_xlabel('Aliased Ch.')
    ax1.set_ylabel('Reproduced Ch.')
    
    ax1.tick_params(axis="y", direction="out")
    ax1.tick_params(axis="x", direction="out")
    ax2.tick_params(axis="y", direction="out")
    ax2.tick_params(axis="x", direction="out")
    ax3.tick_params(axis="y", direction="out")
    ax3.tick_params(axis="x", direction="out")
    
    ax1.minorticks_off()
    ax2.minorticks_off()
    ax3.minorticks_off()
    
    fig.tight_layout()
    
    # colorbar
    fig2 = plt.figure(figsize=(6.6, 1.35))
    fig2.gca().set_visible(False) 
    cbar = fig2.colorbar(im,  orientation='horizontal')
    cbar.set_label(r'A (dB)', rotation=0)
    plt.tight_layout()
    
    if save:
        label = chr(ord('a') + idx)  # 'a', 'b', 'c', ...
        pdf = '.pdf'
        fig.savefig(saveFolder / f'figure2{label}{pdf}')
        fig2.savefig(saveFolder / f'figure2_colorbar{pdf}')
        
