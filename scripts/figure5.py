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

#%% function to read in microphone array transfer functions

def loadMicArrayTFs(azimuths, loadFolder):
    '''
    Load measurements comprising of transfer functions of a linear uniformly 
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
    data = np.load(loadFolder / 'micarray_3m_tfs_comp.npz')
    array = data['arrayTFs']
    angles = data['azimuths']
    
    # Uses just the first 8 microphones as they are scanned over 360 degs
    phi = np.zeros((8, 360, azimuths.shape[0], array.shape[2]), dtype=complex)
    
    for a, angle in enumerate(np.round(np.rad2deg(azimuths)).astype(int)):
        phi[:,:,a,:] = np.roll(array, angle, axis=1)
    
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

Nfft = 2**13
c = 343 # speed of sound
f = np.linspace(0,24000,int(Nfft/2)+1) # Frequency vector

dataFolder = repoRoot / "data" # Folder containing the microphone array data

Na = L - N - 1 # order up to which there is no aliasing


# az_T_list is a list, with an entry for each part of the figure
# each entry is an array of azimuthal angles corresponding to incident sources to make a combined soundfield
az_T_list = []

# Fig 5a, single incident plane wave at 0 deg
az_T_list.append(np.array([0]))  

# Fig 5b, 6 equiangularly spaced incident plane waves
resolution = np.deg2rad(60)
num_steps = int(np.round((2*np.pi)/resolution)+1)
az_T_list.append(np.linspace(-np.pi, np.pi, num_steps, endpoint=True)[:-1])

#%% Decoders and reproduced energy

Y = circHarm(az=phi, N=N, kind=CHdef).T # basis functions for the loudspeaker array sampling positions
D = np.linalg.pinv(Y) # Pseudoinverse decoder
micTF, xCoords, yCoords = loadMicArrayTFs(azimuths=phi, loadFolder=dataFolder) # Transfer functions from loudspeakers to microphones across whole soundfield

#%% 
for idx, az_T in enumerate(az_T_list): # loop over all combinations of soundfields (fig 5a, 5b)
    
    E_R_master = np.zeros((8, 360, 4097))
    E_master   = np.zeros((8, 360, 4097))    
    for indx, currAz in enumerate(az_T): # loop over all individual plane wave sources that sum to make a combined soundfield
        
        currAz = np.array([currAz])
        hoaSignals_T = circHarm(az=currAz, N=N, kind=CHdef).T # the target HOA signals for a source at direction az_T
            
        # Loudspeaker gains to reproduce source at az_T using decoder d
        g = np.zeros((L,len(currAz)))   
        for i in range(len(currAz)):
            g[:,i] = np.dot(D,hoaSignals_T[:,i]) 

        # Simulate the soundfield at the microphone positions for the given loudspeaker gains
        p = np.zeros((micTF.shape[0], micTF.shape[1], micTF.shape[3]), dtype=complex)
        for i in range(len(f)):
            for ii in range(micTF.shape[0]):
                p[ii,:,i] = micTF[ii,:,:,i] @ np.squeeze(g)
        E_R = np.square(np.abs(p))
        
        # reference soundfield for a pure incident plane wave
        refTF, _, _ = loadMicArrayTFs(azimuths=currAz, loadFolder=dataFolder)
        refTF = np.squeeze(refTF)
        E = np.square(np.abs(refTF))
        
        
        normFactor = 1/(2*np.pi) # 1.4pi for 3D, 1/2pi for 2D
        E *= normFactor
        E_R *= normFactor
                
        
        #### Output the energies
        E_R_master += E_R
        E_master += E
    
    
    #%% Plot the energy of the reproduced soundfield
    
    
    plotFreq = 3000
    plotSpkandSrc = False
    rasterized = True
      
    fIndx = np.abs(f - plotFreq).argmin()
    print('Analysis frequency is ' + str(f[fIndx]) + ' Hz')
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)  
    
    # Plot the loudspeaker positions
    if plotSpkandSrc:
        for l in range(len(phi)):
            xplot = np.amax(xCoords) * np.sin(-1*phi[l])
            yplot = np.amax(xCoords) * np.cos(-1*phi[l])
            ax1.plot( xplot, yplot, 'ks', markersize=2.5) # Loudspeaker squares
        # Plot the indicident source direction(s)
        for currAz in az_T:
            ax1.plot( np.amax(xCoords) * np.sin(-1*currAz), np.amax(xCoords) * np.cos(-1*currAz), 'ro', markersize=2.5)
    
    vmin = -10
    vmax = 0  
    norm = abs(E_master[7,0,fIndx])
    plotData = 10*np.log10( E_R_master[:,:,fIndx] / norm )
    plotData = np.roll(plotData, 90, axis=1) # apprantely i have to roll the data by 90 degrees...
    plotData = plotData.flatten()
    pc1 = plt.tricontourf(xCoords.flatten(), yCoords.flatten(), plotData, np.linspace(vmin,vmax,100),  extend='both', cmap='viridis', rasterized=rasterized)
   
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_xlim(-0.26, 0.26)
    ax1.set_ylim(-0.26, 0.26)
    
    cbar1 = fig1.colorbar(pc1)  
    cbar1.set_label(r'Energy (dB)', rotation=90)
    cbar1.set_ticks(np.arange(vmin, vmax+5, 2.5))
    
    # Add an N=Kr circle
    circle = plt.Circle((0, 0), N/(2*np.pi*f[fIndx]/c), color='r', linewidth='1', linestyle='--', fill=False)
    ax1.add_artist(circle)
    circle = plt.Circle((0, 0), Na/(2*np.pi*f[fIndx]/c), color='r', linewidth='1', linestyle='--', fill=False)
    ax1.add_artist(circle)
    
    plt.axis('scaled')  
    plt.tight_layout()
    
    if save:
        label = chr(ord('a') + idx)  # 'a', 'b', 'c', ...
        pdf = '.pdf'
        fig1.savefig(saveFolder / f'figure5{label}{pdf}')
        
