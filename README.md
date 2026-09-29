# Determinants of Groundwater Fauna Occurrence in Central European Cities

This repository accompanies the manuscript: 
Meyer, L., Hemmerle, H., Becher, J., Englisch, C., Griebler, C., Bayer, P. (202X): Determinants of Groundwater Fauna Occurrence in Central European Cities.

The scripts assess the relevance of individual environmental variables to the presence of stygobiont crustaceans in urban groundwater, using classification
models with iterative feature elimination based on permutation importance.


## Repository contents

| File / folder | Description |
|---|---|
| `README.md` | Description of the repository |
| `LICENSE` | License of the code (MIT) |
| `LICENSE-DATA` | License of the data (CC BY 4.0) |
| `requirements.txt` | Python package versions used |
| `dataset.csv` | Input data (one row per sample, see column description below) |
| `01_model_classification.py` | Model comparison with iterative feature elimination; writes the result tables |
| `02_summary_figures.py` | Creates the figures from the result tables |
| `results/tables/<REGION>/` | Result tables underlying the published figures: `MODEL_*.csv` (cross-validated accuracy of all models per elimination step) and `PERIM_*.csv` (permutation importance per elimination step) |
| `results/figures/` | Figures created by `02_summary_figures.py` |

Regions: `HAL` = Halle (Saale), `MUC` = Munich, `VIE` = Vienna, `AGG` = all regions aggregated.


## Data

| Column | Description | Unit |
|---|---|---|
| `ID` | Sample ID | - |
| `REG` | Region | - |
| `DATE` | Sampling date | - |
| `Target` | Stygobiont crustaceans present (1 = yes, 0 = no) | - |
| `VAL_CHL` | Chloride | mg/l |
| `VAL_DSW` | Distance to surface water | m |
| `VAL_GWD` | Depth to groundwater | m |
| `VAL_K` | Hydraulic conductivity | m/s |
| `VAL_SUS` | Surface sealing | % |
| `VAL_NH4` | Ammonium | mg/l |
| `VAL_NO3` | Nitrate | mg/l |
| `VAL_DOC` | Dissolved organic carbon | mg/l |
| `VAL_DO` | Dissolved oxygen | mg/l |
| `VAL_PO4` | Phosphate | mg/l |
| `VAL_TMP` | Water temperature | °C |

All columns starting with `VAL_` are used as predictors.

## Method (short summary)

- **Models:** logistic regression, SVM, k-nearest neighbours, decision tree, random forest, gradient boosting, AdaBoost, LDA and a multilayer perceptron. Hyperparameters are tuned with `GridSearchCV`.
- **Validation:** stratified k-fold cross-validation (5 folds per city, 10 folds for the aggregated data set) with accuracy as the metric.
- **Baseline:** a dummy classifier that always predicts the most frequent class. Only models that outperform this baseline are considered relevant.
- **Feature elimination:** permutation importance (10 repeats per fold) is averaged across all relevant models. The least important predictor is removed and the procedure is repeated until one predictor remains or no model outperforms the baseline anymore.

## Software

The analysis was implemented in Python using scikit-learn (Pedregosa et al., 2011),
pandas (McKinney, 2010), NumPy (Harris et al., 2020) and Matplotlib (Hunter, 2007).
Package versions are listed in `requirements.txt`.

### References

- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
- McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

## License

Code: MIT © 2026 
Data: CC BY 4.0 

## Contact

Laura Meyer, laura.meyer@geo.uni-halle.de. 
