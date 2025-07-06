# The Ring of Silence in Ambisonics and Binaural Audio Reproduction

**Authors**: Jacob Hollebon, Filippo Maria Fazi  
**Affiliation**: Institute of Sound and Vibration Research (ISVR), University of Southampton  
**Year**: 2025

## Overview

This repository contains Python scripts that support the analyses, simulations, and visualizations presented in the paper:

**"The Ring of Silence in Ambisonics and Binaural Audio Reproduction"**

Each script corresponds to performing the relevant simulations and plots for a specific figure in the paper. 

## Structure

```
.
├── LICENSE
├── README.md
├── data/ # Accompanying data required for the simulations
├── scripts/ # Python scripts to generate each figure
│ ├── figure1.py
│ ├── figure2.py
│ └── ...
└── figures/ # Output folder for saved figures and data

```

## Requirements

- Python
- `numpy`, `matplotlib`, `scipy`, `scienceplots`, `hos`
- The HOS python toolbox is used for core spherical/circular harmonic operations and can be found here https://github.com/jacobhollebon/hos/

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

2. Run the desired script:
   ```bash
   python figure_1.py
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

```
@article{hollebon2025ring,
  title     = {The Ring of Silence in Ambisonics and Binaural Audio Reproduction},
  author    = {Filippo Maria Fazi and Jacob Hollebon},
  journal   = {<Journal/Conference Name>},
  year      = {2025},
  url       = {<DOI or URL>}
}
```

## Contact

For questions or collaborations:  
j.hollebon@soton.ac.uk | filippo.fazi@soton.ac.uk 
(ISVR, University of Southampton)
