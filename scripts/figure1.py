# -----------------------------------------------------------------------------
# Script for the paper:
# "The Ring of Silence in Ambisonics and Binaural Audio Reproduction"
#
# Authors: Jacob Hollebon, Filippo Maria Fazi
#
# Figure 1
# Description: Aliasing analysis of a circular loudspeaker array performing 2D Ambisonics
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

#%% User parameters
CHdef = 'real' # 'real' or 'exp': type of circular harmonics

N = 2 # Order of reproduction
Na = 6 # Order of the loudspeaker array
Ndash = 13 # Order up to which aliasing is evaluated

L = (2*Na)+1 # Number of loudspeakers, max number possible
samplingPositions = np.arange(0, 2*np.pi, (2*np.pi/L))

#%% Plot of the loudspeaker layout

# Speaker positions in x, y
Lpos = np.zeros((2,L))
r = 1 # radius of the loudspeaker array
for i in range(L):
    Lpos[0,i] = r*np.cos(samplingPositions[i])
    Lpos[1,i] = r*np.sin(samplingPositions[i])

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.gca().set_aspect('equal', adjustable='box')
ax1.arrow(-1.2, 0, 2.4, 0, width=0.002, length_includes_head=True, head_width=0.05, edgecolor='k', facecolor='k')
ax1.arrow(0, -1.2, 0, 2.4, width=0.002, length_includes_head=True, head_width=0.05, edgecolor='k', facecolor='k')
ax1.scatter(Lpos[0,:], Lpos[1,:], s=100, color='m', marker='s')
ax1.set_xlabel(r'y'); ax1.xaxis.set_label_position("top")
ax1.set_ylabel(r'x',rotation=90); ax1.yaxis.set_label_position("right")
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, which='minor', linestyle='--') 
ax1.minorticks_on()
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.tick_params(which='both', length=0)
ax1.spines['top'].set_color('none')
ax1.spines['bottom'].set_color('none')
ax1.spines['left'].set_color('none')
ax1.spines['right'].set_color('none')
plt.tight_layout()


#%% Aliasing matrix

# CHs at speaker sampling positions up to order of array
Y1 = circHarm(samplingPositions, N, kind=CHdef) 
# CHs at speaker sampling positions up to order of aliasing matrix evaluation
Y2 = circHarm(samplingPositions, Ndash, kind=CHdef)

Y1pinv = np.linalg.pinv(Y1) # Pseudoinverse decoder
E = np.abs( Y1pinv @ Y2  ) 
E[E < 0.000001] = 0.000001 # clipping to avoid log10(0)


#%% Plot of the aliasing matrix

 # Hlines for each order up to N
maxN_hlines = int(N)
n_hlines = np.arange(0,maxN_hlines+1)
ACN_hlines = (n_hlines*2)-0.5 +1
numCoeffsY2 = Y2.shape[1]
# Vlines for each order up to Ndash
maxN_vlines = int(Ndash)
n_vlines = np.arange(0,maxN_vlines+1)
ACN_vlines = (n_vlines*2)-0.5 +1
numCoeffsY1 = Y1.shape[1]

fig = plt.figure(figsize=(3.3, 1.35)) 
ax1 = fig.add_subplot(1, 1, 1)
im = ax1.imshow(20*np.log10(abs(E)), vmax=0, vmin=-60, aspect="auto", cmap='Greys_r')
ax1.hlines(ACN_hlines, -0.5, numCoeffsY2, 'w')
ax1.vlines(ACN_vlines, -0.5, numCoeffsY1, 'w')
ax1.set_ylim(-0.5, numCoeffsY1-0.5)
ax1.set_xlim(-0.5, numCoeffsY2-0.5)

xticks = list(range(0,28,4))
ax1.set_xticks(xticks)
ax1.set_yticks([0,4])
ax1.minorticks_off()

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
    
orderticksaliased = (ACN_vlines+0.5)/(2*Ndash +1)
ax2.set_xticks(orderticksaliased[:-1])
ax2.set_xticklabels(orderlabelsaliased[1:])
ax2.set_xlabel("Order")

orderticksrep = (ACN_hlines+0.5)/(2*N +1)
ax3.set_yticks(orderticksrep[:-1])
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
fig2 = plt.figure()
fig2.gca().set_visible(False) 
cbar = fig2.colorbar(im,  orientation='horizontal')
cbar.set_label(r'A (dB)', rotation=0)
plt.tight_layout()


if save:
    pdf = '.pdf'
    fig.savefig(saveFolder / f'figure1{pdf}')
    fig2.savefig(saveFolder / f'figure1_colorbar{pdf}')
    

 

    