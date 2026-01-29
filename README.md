# HyperKamiokande Neutrino Event Analysis

This project focuses on the analysis and energy reconstruction of neutrino events for the HyperKamiokande experiment.

## Project Structure

- **`data_processing/`**: Scripts for data loading and cleaning.
- **`models/`**: Regression models (Linear, Cyclic) for energy reconstruction.
- **`visualization/`**: Plotting utilities.
- **`notebooks/`**:
  - `01_Energy_Prefit_Pipeline.ipynb`: **Quantitative Mode**. Starts with exploratory stats relevant to modeling, then trains prediction models (Linear, Cyclic) and analyzes energy reconstruction residuals.
  - `02_Charge_And_Geometry_Analysis.ipynb`: **Qualitative/Deep Dive**. Focuses on detailed event inspection, 3D visualization, incidence angle calculations, and charge distribution analysis.

## Usage

1.  **Quantitative Analysis**: Run `01_Energy_Prefit_Pipeline.ipynb` first to establish the energy reconstruction baseline and pipeline.
2.  **Qualitative Detail**: Run `02_Charge_And_Geometry_Analysis.ipynb` to investigate specific event geometries and detector responses (angles, charge maps).
