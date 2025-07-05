# Data Folder README

This folder contains data files used in this project.

---

## Files and Descriptions

- **ku100_2702.npz**  
  Measurement set `HRIR_2702` from reference [1].  
  Contains arrays:  
  - `hrir`: shape `(samplingPositions, ears, samples)`  
  - `grid`: shape `(samplingPositions, (azimuth, elevation, radius))`

- **ku100_circular.npz**  
  Measurement set `HRIR_CIRC360` from reference [1].  
  Contains arrays:  
  - `hrir`: shape `(samplingPositions, ears, samples)`  
  - `azimuths`: shape `(azimuth)`

- **micarray_3m.npz**  
  Measurements using a linear microphone array of an anechoic loudspeaker positioned in the far field measured across 360 degrees incident azimuths.
  Contains arrays:  
  - `arrayTFs`: shape `(microphones, azimuths, freqs)`  
  - `azimuths`: shape `(azimuth)`

- **spherical_O6_84pt_t12.npy**  
  Sampling grid data for t-design t 12, 84 points.  
  Shape: `((azimuth, elevation, radius), samplingPositions)`

- **spherical_O6_equalarea.npy**  
  Equal-area sampling grid, 49 points.  
  Shape: `((azimuth, elevation, radius), samplingPositions)`

---

## References

[1] Bernschütz, B. (2013). *A Spherical Far Field HRIR / HRTF Compilation of the Neumann KU 100*. Proceedings of the 39th DAGA, Meran (Italy).  
DOI: [10.5281/zenodo.3928296](https://doi.org/10.5281/zenodo.3928296)  
License: Creative Commons Attribution 3.0 Unported (CC BY 3.0)  
[Original Data on Zenodo](https://zenodo.org/record/3928297)
