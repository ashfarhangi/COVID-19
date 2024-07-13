
<div align="center">

# DemandNet: A Novel Deep Learning Model for Hotel Demand and Revenue Prediction amid COVID-19

[![Conference](http://img.shields.io/badge/HICSS-2022-4b44ce.svg)](https://arxiv.org/abs/2203.04383)
</div>



## Overview

The COVID-19 pandemic has significantly impacted the tourism and hospitality sector, with public policies such as travel restrictions and stay-at-home orders affecting tourist activities and business operations. To address this, we developed DemandNet, a novel deep learning framework for predicting time series data under the influence of the COVID-19 pandemic. DemandNet aims to support managerial and organizational decision-making by providing accurate and interpretable forecasts.

## Key Features

- **Feature Selection**: A mechanism to select the top static and dynamic features embedded in the time series data.
- **Nonlinear Modeling**: A multilayer neural network that provides interpretable insights into previously seen data.
- **Robust Predictions**: A prediction model leveraging selected features and nonlinear models to make robust long-term forecasts.
- **Dynamic Dropout Optimization**: Minimizes prediction uncertainties and provides optimal confidence in forecasts.

## Contributions

1. **Feature Selection Mechanism**: Selects the top static and dynamic features of a time series, enhancing the ability to capture complex critical features.
2. **Multilayer Neural Network**: Derives the nonlinear relationship of selected features to the predictor, providing interpretable insights.
3. **Novel Prediction Model**: Leverages a dynamic dropout optimization mechanism for robust multi-step time series prediction.
4. **Capability for New Data**: Capable of predicting newly added time series data without previous training.


A repository for COVID-19 factors and impacts on US economy.
To get a local copy up and running follow these simple example steps.

### Datasets

Gathered State-level data: 

![](https://raw.githubusercontent.com/ashfarhangi/COVID-19/main/visualization/heat-map.png)


loc: data/COVID19_state.xlsx



[![Star History Chart](https://api.star-history.com/svg?repos=ashfarhangi/COVID-19&type=Date)](https://star-history.com/#ashfarhangi/COVID-19&Date)

### Prerequisites

- Tensorflow 2.0.2


### Installation

1. Clone the repo

   ```Python
   git clone https://github.com/ashfarhangi/COVID-19.git
   ```

2. Install requirement packages

   ```Python
   pip install -r requirements.txt
   ```

3. Run model.py 


# Citation 

```bibtex
@inproceedings{farhangidemand,
  title={A Novel Deep Learning Model For Hotel Demand and Revenue Prediction amid COVID-19},
  author={Farhangi, Ashkan and Huang, Arthur and Guo, Zhishan},
  booktitle={Proceedings of the 55th Hawaii International Conference on System Sciences (HICSS 2022)},
  year={2022},
  organization={HICSS-55}
}
```