# EEGTool

EEGTool is a comprehensive toolbox designed for the analysis and visualization of Electroencephalography (EEG) data. It provides a user-friendly interface and a suite of powerful functions to facilitate various stages of EEG data processing, from raw data import to advanced statistical analysis and visualization.

## Features

*   **Data Import:** Supports various EEG data formats (e.g., .set, .edf, .bdf).
*   **Preprocessing:** Includes functionalities for filtering, artifact rejection (e.g., ICA), and re-referencing.
*   **Event-Related Potentials (ERPs):** Tools for epoching, baseline correction, and ERP averaging.
*   **Time-Frequency Analysis:** Capabilities for computing power spectra and time-frequency representations.
*   **Source Localization:** (Planned) Integration with external toolboxes for source reconstruction.
*   **Statistical Analysis:** Built-in functions for statistical comparisons and hypothesis testing.
*   **Visualization:** Extensive plotting options for raw data, ERPs, topographies, and time-frequency maps.
*   **Batch Processing:** Scripting capabilities for automated analysis of multiple datasets.
*   **User Interface:** Intuitive graphical user interface (GUI) for easy navigation and operation.

## Getting Started

### Prerequisites

*   EEGLAB Toolbox (required for core EEG functionalities) - MATLAB

### Installation

1.  **Download EEGTool:** Clone or download the EEGTool repository from GitHub.
2.  **Add to MATLAB Path:** Open MATLAB, navigate to the EEGTool directory, and add it to your MATLAB path using `addpath(genpath('your_eegtool_directory'))`. (MATLAB)
3.  **Install EEGLAB:** If you don't have EEGLAB installed, download it from the official EEGLAB website and add it to your MATLAB path as well. (MATLAB)

### Running EEGTool

Once installed, you can launch the EEGTool GUI by typing `EEGTool` in the MATLAB command window and pressing Enter.

## Documentation

Detailed documentation, tutorials, and examples will be provided in the `docs` directory.

## Contributing

We welcome contributions to EEGTool! If you have suggestions, bug reports, or would like to contribute code, please refer to our `CONTRIBUTING.md` file for guidelines.

## License

EEGTool is released under the