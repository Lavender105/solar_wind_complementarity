# Solar and wind complementarity substantially increases renewable energy penetration in China
Yuan Hu, Hou Jiang, Chuan Zhang, Jianlong Yuan, Mengting Zhang, Ling Yao, Qiang Chen, Jichao Wu, Hualong Zhang, Xiang Li, Weiyu Zhang, Congcong Wen, Fan Zhang*, Chaohui Yu*, Yu Liu*

This repository holds codes for the paper entitled "Solar and wind complementarity substantially increases renewable energy penetration in China," currently under review at Nature Energy.

## Abstract
The rapid expansion of solar and wind installations in China has introduced significant power curtailment challenges due to their inherent variability. Understanding the impact of solar and wind energy complementarity in enhancing their penetration is urgently necessary; however, the lack of real arrangement of solar and wind installations has hindered the progress. Here, we present a unified national inventory of 319,972 solar PV installations and 91,609 wind installations in 2022, derived from high-resolution remote sensing imagery and a advanced deep learning pipeline. We identify the optimal energy complementarity strategy for each county and quantified its effectiveness in mitigating power fluctuations.  Additionally, penetration analysis reveals that nationwide inter-provincial energy collaboration can enhance solar and wind energy absorption by 101.07 TWh in a 100% flexible system, equivalent to 9.8% of their total generation or 121 hours of national average load. Our study provides critical insights for developing policies on penetration mechanisms.

## Contents
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](#license)

## System Requirements
### Hardware requirements
The codes requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements
#### OS Requirements
The codes has been tested on the following systems:

- macOS: Sonoma 14.5
- Linux: Ubuntu 20.04

#### Python Dependencies
The package dependencies are in requirements.txt.

## Installation Guide
Follow these steps to set up the development environment:

### Using pip (requirements.txt)

1. Clone the repository:
   ```bash
   git clone https://github.com/Lavender105/solar_wind_complementarity.git
   cd solar_wind_complementarity
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
### Using conda (environment.yml)
1.	Clone the repository:
   ```bash
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

