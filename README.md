# The Ring of Silence — Code Repository

**Authors**: Filippo Maria Fazi, Jacob Hollebon  
**Affiliation**: Institute of Sound and Vibration Research (ISVR), University of Southampton

## Overview

This repository contains Python scripts supporting two related publications on the "Ring of Silence" phenomenon in Ambisonics and binaural audio reproduction.

### IEEE TASLP Paper (Primary)

**"The Ring of Silence in Ambisonics: Spectral Impairments in Loudspeaker and Binaural Reproduction"**  
F. M. Fazi and J. Hollebon — *IEEE Transactions on Audio, Speech and Language Processing*, vol. 34, pp. 1061–1071, 2026  
DOI: [10.1109/TASLPRO.2026.3655624](https://doi.org/10.1109/TASLPRO.2026.3655624)

Scripts and figures in `scripts/` and `figures/`.

### AVARIG 2026 Conference Paper (Extension)

**"Investigating the 'Ring of Silence' in Loudspeaker and Binaural Reproduction Using Advanced Ambisonic Decoding Strategies"**  
F. M. Fazi, J. Hollebon, Y. Li — *AES International Conference on Audio for Virtual and Augmented Reality and Immersive Games*, Paris, France, June 30 – July 3, 2026

Scripts and figures in `avarig_scripts/` and `avarig_figures/`.

## Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── data/                  # Shared data for all simulations
├── scripts/               # Scripts for IEEE TASLP paper figures
├── figures/               # Output figures for IEEE TASLP paper
├── avarig_scripts/        # Scripts for AVARIG 2026 paper figures
└── avarig_figures/        # Output figures for AVARIG 2026 paper
```

## Requirements

- Python
- `numpy`, `matplotlib`, `scipy`, `scienceplots`, `hos`
- The HOS python toolbox is used for core spherical/circular harmonic operations and can be found here https://github.com/jacobhollebon/hos/
- A working LaTeX installation is required for rendering plot labels and annotations

Install dependencies via:

```bash
pip install -r requirements.txt
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/jacobhollebon/ring-of-silence.git
   cd ring-of-silence
   ```

2. Run the desired script, e.g. for the IEEE paper:
   ```bash
   python scripts/figure1.py
   ```
   Or for the AVARIG paper:
   ```bash
   python avarig_scripts/figure3a_and_3b_and_3c_and_4.py
   ```

## Third-Party Data Attribution

This repository includes modified data derived from:

**Benjamin Bernschütz**,  
*A Spherical Far Field HRIR / HRTF Compilation of the Neumann KU 100*,  
Proceedings of the 39th DAGA, 2013, pp. 592–595.  
DOI: [10.5281/zenodo.3928296](https://doi.org/10.5281/zenodo.3928296)

Licensed under the [Creative Commons Attribution 3.0 Unported (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/) license.

`data/ku100_circular.npz` corresponds to the `HRIR_CIRC360` measurement set

`data/ku100_2702.npz` corresponds to the `HRIR_2702` measurement set


## License

This work is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.  
See the [LICENSE](./LICENSE) file or visit [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/) for details.


## Citation

If you use this code, please cite the relevant paper(s):

### IEEE TASLP Paper

```
@article{fazi2026ring,
  title   = {The Ring of Silence in Ambisonics: Spectral Impairments in Loudspeaker and Binaural Reproduction},
  author  = {Fazi, Filippo Maria and Hollebon, Jacob},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  volume  = {34},
  pages   = {1061--1071},
  year    = {2026},
  doi     = {10.1109/TASLPRO.2026.3655624}
}
```

### AVARIG 2026 Paper

```
@inproceedings{fazi2026investigating,
  title     = {Investigating the ``Ring of Silence'' in Loudspeaker and Binaural Reproduction Using Advanced Ambisonic Decoding Strategies},
  author    = {Fazi, Filippo Maria and Hollebon, Jacob and Li, Yueheng},
  booktitle = {AES International Conference on Audio for Virtual and Augmented Reality and Immersive Games},
  address   = {Paris, France},
  year      = {2026}
}
```

## Contact

For questions or collaborations:  
j.hollebon@soton.ac.uk | filippo.fazi@soton.ac.uk  
(ISVR, University of Southampton)
