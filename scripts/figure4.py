# -----------------------------------------------------------------------------
# Script for the paper:
# "The Ring of Silence in Ambisonics and Binaural Audio Reproduction"
#
# Authors: Jacob Hollebon, Filippo Maria Fazi
#
# Figure 4b and 4c
# Description: Average energy reproduced by a circular loudspeaker array performing 2D Ambisonics 
# using measured loudspeaker transfer functions
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

#%% function to read in microphone array transfer functions

def loadMicArrayTFs(azimuths, loadFolder):
    '''
    Load measurements comprising of transfer functions of a Linear uniformly 
    spaced microphone array for a loudspeaker source incident at 1 deg intervals
    over 360 degs azimuthal positons.
    
    Measured using a single loudspeaker and rotating the microphone array through 360 degs.

    Returns the TFs for a specified set of incident angles, azimuths
    Parameters
    ----------
    azimuths : array
        Array of incident azimuthal angles in radians to retrieve from the measured dataset.
    loadFolder : str
        Str path to the folder containing the measured dataset.

    Returns
    -------
    phi : array
        Transfer functions for each of the microphones and sampled azimuths.
    xCoords : array
        x coordinates of each of the microphone array elements.
    yCoords : array
        y coordinates of each of the microphone array elements.

    '''
    data = np.load(loadFolder / 'micarray_3m.npz')
    array = data['arrayTFs']
    angles = data['azimuths']
    
    
    # UseS just the first 8 microphones as they are scanned over 360 degs
    phi = np.zeros((8, 360, azimuths.shape[0], array.shape[2]), dtype=complex)
    
    for a, angle in enumerate(np.round(np.rad2deg(azimuths)).astype(int)):
        phi[:,:,a,:] =  np.roll(array, angle, axis=1)[:8, :, :]
    
    micSpacing = 0.03714 # in metres
    ID         = np.arange(7,-1,-1) 
    micRadius  = ID * micSpacing
    
    anglesScanned = np.deg2rad(angles[:360])
    
    xCoords = micRadius[:,np.newaxis] * np.cos(anglesScanned)
    yCoords = micRadius[:,np.newaxis] * np.sin(anglesScanned)
    
    return phi, xCoords, yCoords


#%% User setup
CHdef = 'real' # 'real' or 'exp': type of circular harmonics

N  = 2 # order of the b-format input signal
N_ = 6  # order of the loudspeaker array

L  = 2*N_+1 # number of speakers in array
phi = (np.arange(L)+0.5) *2*np.pi/L # Loudspeaker positions, uniform sampling, add 0.5 to avoid speaker at 0 degrees

az_T = np.deg2rad(np.array([ 13.84615385])) # incident plane wave azimuthal angle

Nfft = 2**13
c = 343 # speed of sound
f = np.linspace(0,24000,int(Nfft/2)+1) # Frequency vector

dataFolder = repoRoot / "data" # Folder containing the microphone array data

#%% Decoders and reproduced energy

hoaSignals_T = circHarm(az=az_T, N=N, kind=CHdef).T # the target HOA signals for a source at direction az_T

Y = circHarm(az=phi, N=N, kind=CHdef).T # basis functions for the loudspeaker array sampling positions
D = np.linalg.pinv(Y) # Pseudoinverse decoder

# Loudspeaker gains to reproduce source at az_T using decoder d
g = np.zeros((L,len(az_T)))   
for i in range(len(az_T)):
    g[:,i] = np.dot(D,hoaSignals_T[:,i]) 

Na = L - N - 1 # order up to which there is no aliasing

#%% Reproduced energy by the loudspeaker array using measured microphone transfer functions to sample the soundfield

# Simulate the soundfield at the microphone positions for the given loudspeaker gains
micTF, xCoords, yCoords = loadMicArrayTFs(azimuths=phi, loadFolder=dataFolder)
p = np.zeros((micTF.shape[0], micTF.shape[1], micTF.shape[3]), dtype=complex)
for i in range(len(f)):
    for ii in range(micTF.shape[0]):
        p[ii,:,i] = micTF[ii,:,:,i] @ np.squeeze(g)

E_R = np.sum(np.square(np.abs(p)), axis=1)

# reference soundfield for a pure incident plane wave
refTF, _, _ = loadMicArrayTFs(azimuths=az_T, loadFolder=dataFolder)
refTF = np.squeeze(refTF)
E = np.sum(np.square(np.abs(refTF)), axis=1)


normFactor = 1/(2*np.pi) # 1.4pi for 3D, 1/2pi for 2D
E *= normFactor
E_R *= normFactor

#%% Plots of the reproduced soundfield at the microphone positions

micSpacing = 0.03714 # in metres
ID         = np.arange(7,-1,-1) 
micRadius  = ID * micSpacing
ylims = [-15,5]
fLoop = [3000, 4000] # Frequencies to produce plots of the soundfield at

  
for idx, ii in enumerate(fLoop):
    fIndx = np.abs(f - ii).argmin()
    print('Analysis frequency is ' + str(f[fIndx]) + ' Hz')
    
    norm = abs(E[0,fIndx])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    
    ax1.plot(micRadius,10*np.log10( E[:,fIndx] /norm), color='C0', label='Reference')
    ax1.plot(micRadius,10*np.log10( E_R[:,fIndx] /norm), color='C2', ls=':', label= 'Reproduced')

    k = (2*np.pi*f[fIndx]) / c
    r = N / k
    ra = (Na+1)/k
    ax1.vlines(r, ylims[0], ylims[1], color='0.1', ls='--')
    ax1.vlines(ra, ylims[0], ylims[1], color='0.1', ls='--')
    
    ax1.set_xlabel(r'r');
    ax1.set_ylabel(r'SPL (dB)')
    ax1.set_ylim(ylims[0],ylims[1])
    ax1.set_xlim(0,0.259)
    ax1.set_yticks([-15,-10,-5,0,5])
    ax1.grid(True, which='major', linestyle='-')
    ax1.grid(True, which='minor', linestyle='--') 
    ax1.minorticks_on()
    leg = ax1.legend(loc=3, framealpha=0.7, facecolor='white', frameon=True)
    leg.get_frame().set_linewidth(0.0)
    fig1.tight_layout()
    fig1.show()
    if save:
        label = chr(ord('b') + idx)  # 'a', 'b', 'c', ...
        pdf = '.pdf'
        fig1.savefig(saveFolder / f'figure4{label}{pdf}')
        
