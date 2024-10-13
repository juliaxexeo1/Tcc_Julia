
# WALE-a

***WALE-a: A Multilevel Wavelet Decomposition Method to Classify Time Series.***

> <div align="justify">The main inspiration for our algorithm was the layered architecture of CNN networks. In CNN networks, input data undergoes transformations throughout the network until classification. Similarly, our algorithm has four layers: the first three layers handle data transformations, and the last layer performs classification. Our input data consists of events in raw time series format. For data transformations, we employed wavelet transformations and statistical metrics. We chose discrete wavelet transform (DWT) over continuous wavelet transform (CWT) due to its suitability for real-time applications. DWT decomposes the signal into multiple resolution subsets of coefficients by applying high-pass and low-pass filters, resulting in detailed (cD) and approximation (cA) coefficient subsets. This process is repeated until the desired level of decomposition is achieved.</div>


## Wavelets

The WALE-a implementation was developed from [pywavelets](https://pywavelets.readthedocs.io/en/latest/).

## Theories and Datasets

All theoretical basis, tools and datasets for time series can be found in the [TimeseriesClassification project](https://www.timeseriesclassification.com/).

[Datasets](https://www.timeseriesclassification.com/dataset.php)

[Algorithms](https://www.aeon-toolkit.org/en/latest/)

## Codes

### [`test_wale_v1.py`](test_wale_v1.py) - Training and testing WALE-a with a dataset
### [`lib_all_data_analysis_v3.py`](lib_all_data_analysis_v3.py) - Set of functions for WALE-a operation
### [`test_rocket_v2.py`](test_rocket_v2.py) - Training and testing ROCKET classifier with a dataset
### [`rocket_functions.py`](rocket_functions.py) - Functions to the ROCKET classifier


## Requirements

* Python 3.10;
* Numba;
* NumPy;
* scikit-learn (or equivalent);
* sktime 0.15 or aeon;
* Pywavelets 1.4.1;


