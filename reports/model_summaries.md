# Model Training Summary

## Overall Model Comparison
| Model | Recall |
|-------|----------------|
| logistic_regression | 0.7729 |
| sgdclassifier | 0.8011 |
| linearsvc | 0.7751 |
| gradientboostingclassifier | 0.7959 |
| votingclassifier | 0.7954 |

## Logistic Regression
- **Best Recall**: `0.7729`
- **Best Parameters**:
```python
{'C': 0.01, 'penalty': 'l1'}
```

## Sgdclassifier
- **Best Recall**: `0.8011`
- **Best Parameters**:
```python
{'alpha': 0.0001, 'eta0': 0.01, 'learning_rate': 'optimal', 'loss': 'modified_huber', 'max_iter': 1000, 'penalty': 'elasticnet'}
```

## Linearsvc
- **Best Recall**: `0.7751`
- **Best Parameters**:
```python
{'C': 0.01, 'loss': 'squared_hinge', 'max_iter': 15000, 'penalty': 'l1', 'tol': 0.0001}
```

## Gradientboostingclassifier
- **Best Recall**: `0.7959`
- **Best Parameters**:
```python
{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 150, 'subsample': 0.8}
```

## Votingclassifier
- **Best Recall**: `0.7954`
- **Best Parameters**:
```python
{'voting': 'soft'}
```

---

### Best Model: `sgdclassifier` with **recall** = `0.8011`

#### Classification Report on entire test set
```
              precision    recall  f1-score   support

         0.0       0.76      0.70      0.73     27299
         1.0       0.73      0.78      0.75     28133

    accuracy                           0.74     55432
   macro avg       0.74      0.74      0.74     55432
weighted avg       0.74      0.74      0.74     55432

```