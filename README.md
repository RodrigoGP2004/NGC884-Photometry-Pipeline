# NGC884-Photometry-Pipeline
Complete astronomical data reduction and analysis pipeline developed in Python, customized for a set of observations of the open cluster NGC 884. The project processes raw observational FITS data, performs aperture photometry, and derives its fundamental astrophysical properties, including the Hertzsprung-Russell (H-R) diagram and the Initial Mass Function (IMF).

## Pipeline

The project is divided into two main modules:

### 1. Data Reduction & Photometry (`src/scriptreduction.py`)
- **Calibration:** Automated creation of Master Bias, Master Dark, and Master Flats. Science image calibration.
- **Image Alignment:** Advanced source matching using `scipy.spatial.KDTree` and affine transformations via `RANSAC` (`skimage.transform`) to align different filter exposures (B and V).
- **Aperture Photometry:** Source extraction using `DAOStarFinder` and precise flux measurement with `photutils` (Circular Apertures and Background Annuli).
- **Photometric Calibration:** Cross-matching local catalogs with Gaia DR3 data using `GaiaXPy` to calibrate instrumental magnitudes into the standard Johnson-Cousins photometric system.

### 2. Astrophysical Analysis (`src/dataanalysis.py`)
- **Isochrone Fitting:** Interpolation of theoretical isochrones to estimate the cluster's age and distance modulus.
- **Stellar Mass Interpolation:** Mapping observed V magnitudes to stellar masses using `scipy.interpolate.interp1d`.
- **Initial Mass Function (IMF):** Calculation of the stellar density distribution ($dN/dM$). The pipeline fits a power-law to the high-mass end, yielding an index of **$\alpha = 2.26$** (highly consistent with the theoretical Salpeter IMF of $\alpha = 2.35$).
- **Extrapolation Models:** Implementation of Kroupa IMF multi-part power laws to predict the expected number of hidden populations (brown dwarfs, rogue planets) and missing massive stars.

## Resulting plots

### Hertzsprung-Russell (Color-Magnitude) Diagram
Comparison of the reduced observational data (taken during a bad sight night) against Vizier catalogs and theoretical isochrones ($\sim 12 - 15$ Myr).
![HR](plots/HRFinal3.png)

### Initial Mass Function (IMF)
The derived mass distribution from our observations. The red line shows our empirical fit ($\alpha = 2.26$), closely following the Salpeter theoretical slope (dashed blue).
![IMF Fit](plots/IMF.png)

![Mass_histogram](plots/Mass_histogram.png)
