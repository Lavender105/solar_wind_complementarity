# Advancing solar and wind penetration in China via energy complementarity
Yuan Hu, Hou Jiang, Chuan Zhang, Jianlong Yuan, Mengting Zhang, Ling Yao, Qiang Chen, Jichao Wu, Hualong Zhang, Xiang Li, Weiyu Zhang, Congcong Wen, Fan Zhang*, Chaohui Yu*, Yu Liu*

This repository holds codes for the paper entitled "Advancing solar and wind penetration in China via energy complementarity," currently under review at Nature.

## Abstract
The rapid expansion of solar and wind installations in China has introduced significant power curtailment challenges due to their inherent variability. Understanding the impact of solar and wind energy complementarity in enhancing their penetration is urgently necessary; however, the lack of real arrangement of solar and wind installations has hindered the progress. Here, we present a unified national inventory of 319,972 solar PV installations and 91,609 wind installations in 2022, derived from high-resolution remote sensing imagery and a advanced deep learning pipeline. We identify the optimal energy complementarity strategy for each county and quantified its effectiveness in mitigating power fluctuations.  Additionally, penetration analysis reveals that nationwide inter-provincial energy collaboration can enhance solar and wind energy absorption by 101.07 TWh in a 100% flexible system, equivalent to 9.8% of their total generation or 121 hours of national average load. Our study provides critical insights for developing policies on penetration mechanisms.

## Contents
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Code Description](#code-description)
- [Demo](#demo)
- [License](#license)

## System Requirements
### Hardware requirements
The code requires a standard computer with enough RAM to support the in-memory operations.

### Software requirements
#### OS Requirements
The codes has been tested on the following systems:

- macOS: Sonoma 14.5
- Linux: Ubuntu 20.04

#### Python Dependencies
All package dependencies are listed in the requirements.txt file.

## Installation Guide
Follow these steps to set up the development environment:

### Using pip (requirements.txt)

1. Clone the repository:
   ```
   git clone https://github.com/Lavender105/solar_wind_complementarity.git
   cd solar_wind_complementarity
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
### Using conda (environment.yml)
1. Clone the repository:
   ```
   git clone https://github.com/Lavender105/solar_wind_complementarity.git
   cd solar_wind_complementarity
   ```
2. Create a conda environment from the environment.yml file:
   ```
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```
   conda activate solarwind
   ```
The installation process should take approximately 40 seconds on a standard recommended computer.

## Code Description

The code folder contains Python scripts designed to analyze solar and wind power installations in China, evaluate complementary strategies, and visualize the results through various statistical and geographical plots. Below is a brief description of the functionality of each script:

- **fig1_calculate_provincial_statistic.py**

Calculates the total annual power generation for solar and wind installations in each province of China for the year 2022, and generates a bar chart for visualization.

- **fig1_visualize_solar_power_location.py**

Visualizes solar installations across China identified by our deep learning pipeline, and plots a detailed map of the locations.

- **fig1_visualize_wind_power_location.py**
  
Visualizes wind installations across China identified by our deep learning pipeline, and generates a map showing their distribution.

- **fig2_visualize_correlation_map.py**
  
Analyzes different solar-wind complementary strategies, and plots correlation coefficients map under various strategies and the optimal matching maps.

- **fig3_statistic_analysis.py**

Evaluates the effectiveness of the optimal solar-wind complementary strategy in mitigating energy fluctuations, and generates a series of statistical analysis charts and maps.

- **fig4_penetration_analysis.py**
  
Investigates the impact of solar-wind complementarity on energy penetration, and creates related statistical analysis charts and maps.

- **fig5_with_without_storage.py**
  
Analyzes the impact of storage on flexible power generation, and produces corresponding statistical analysis charts and maps.

## Demo
You can run the Python scripts to generate results by following these steps:
1.	Navigate to the code directory:
   ```
   cd code
   ```
2.	Execute the scripts using the python command:
   - **Generate the provincial bar chart for solar and wind power (2022)**
  ```
  python fig1_calculate_provincial_statistic.py
  ```
This script generates a bar chart showing the estimated total solar and wind power generation for 2022. The execution time is approximately 8 seconds.
   - **Visualize solar power installation locations in China**
  ```
  python fig1_visualize_solar_power_location.py
  ```
This script produces a map of solar installations in China. The execution time is approximately 85 seconds.
   - **Visualize wind power installation locations in China**
  ```
  python fig1_visualize_wind_power_location.py
  ```
This script generates a map of wind installations in China. The execution time is approximately 45 seconds.

## License
This project is licensed under the [Apache 2.0 License](LICENSE).
