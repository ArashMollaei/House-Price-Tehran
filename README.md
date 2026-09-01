# Tehran House Price Prediction 

## What is this?
My second end-to-end ML project (regression)

## The problem
Predict house prices in Tehran from 6 features: Area, Room, Parking, Warehouse,
Elevator, Region.
- 3,479 listings scraped from Divar
- Target: Price in USD (right-skewed, median=$97k, max=$3M)
- Business goal: minimize prediction error for typical apartments

## Preprocessing decisions (and why)
| Decision | Why |
|---|---|
| Dropped `Price` (Toman), kept `Price(USD)` | Perfect collinearity; USD avoids confusion |
| Dropped 5 rows with `Price(USD) < $5,000` | Broken records (e.g. $120 "house") |
| Dropped 17 rows with `Area > 500m²` | Inconsistent records (e.g. 863m² with 2 bedrooms) and land/villas (different market) |
| Removed duplicates | Ensured each listing is unique |
| `log1p` transform on target | Right-skewed distribution → train on log, report in dollars via `expm1` |
| Top-30 regions + "Other" | 192 original regions → 31 stable groups; avoids sparse OneHot columns |
| StandardScaler on Area/Room (train only) | Prevents data leakage |

## EDA findings (5 plots)
1. **Target distribution:** Strongly right-skewed (mean=2×median) → log1p confirmed
2. **Outliers:** 310 upper outliers (9%) are legitimate luxury homes, not errors → kept
3. **Price vs Area:** Linear in log space, not raw → Ridge can learn it
4. **Region frequency:** Top 30 cover 70% of data → grouping confirmed
5. **Region signal:** Median prices differ by several times → feature is predictive

## Models compared (with hyperparameter tuning)
| Model | RMSE | MAE |
|---|---|---|
| Ridge (alpha=0.01) | $179k | $63k |
| **RandomForest** (250 trees) | **$113k** | **$50k** |

## Final choice: RandomForest

### Why RandomForest?
RandomForest cannot extrapolate. Ridge in log space predicts extreme values
for houses with Area z-score > 9, then `expm1` amplifies the error
exponentially. Trees predict the average of training samples in each leaf, so
predictions stay within the training range → prevents catastrophic errors.

### Residual analysis
The residual plot shows most predictions are accurate (dense cloud around 0),
but a few predictions are catastrophically wrong (outlier points). The
histogram confirms: ~580 of 694 test samples have residuals near 0, but the
tail extends to −$8.8M.

MAE vs RMSE gap ($50k vs $113k) signals a few catastrophic errors, not uniform 
poor performance.

## What I learned
- Linear models can catastrophically fail on edge cases due to extrapolation
- Tree-based models are more robust to outliers (no extrapolation)
- Filtering inconsistent records before split is data quality, not metric
  manipulation — but the scope must be explicitly defined
- MAE vs RMSE gap signals catastrophic errors on edge cases

## Honest limitations
- **Scope:** Model is only valid for apartments ≤ 500m² (98.7% of listings).
  Larger properties (land/villas) were filtered as data quality issues.
- **Irreducible error:** With only 6 features, within-region variance is the
  limiting factor. MAE=$50k on median=$97k (52%) shows the data doesn't
  contain enough signal for higher precision without additional features
  (floor, age, building quality).

## External benchmark
A similar Kaggle notebook achieved RMSE ≈ $39k using XGBoost, but with a 
narrower scope (Area ≤ 200m², Price ≤ $445k). Direct comparison is invalid 
because the test populations differ.

## Future work
- TargetEncoder for better region signal
- Feature engineering (Area × Region interactions)
- XGBoost/LightGBM
- Geographic features (latitude/longitude)

## Run it
`notebook.ipynb` runs top-to-bottom. The dataset downloads automatically
via `kagglehub`.
