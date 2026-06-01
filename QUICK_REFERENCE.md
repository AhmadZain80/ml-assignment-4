# Quick Reference Guide - Interactive ML Dashboard

## 📋 Quick Start Commands

```bash
# Activate environment
.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Run Jupyter notebook with previews
jupyter notebook GUI_Preview_Dashboard.ipynb

# Run Streamlit app (when created)
streamlit run app.py
```

## 🎨 UI Component Quick Reference

### Sidebar Navigation
- Dataset: Data upload and management
- Algorithms: ML algorithm configuration
- Training: Model training interface
- Visualize: Results visualization dashboard
- Predict: Prediction on new data

### Main Content Areas

| Part | Purpose | Key Elements |
|------|---------|--------------|
| 1 | Data Management | Upload, Preview, Stats, Feature Select |
| 2 | Algorithm Config | kNN, Random Forest, Naive Bayes settings |
| 3 | Training | Train/Test split, Cross-validation, Progress |
| 4 | Visualization | Charts, Feature importance, Comparisons |
| 5 | Prediction | Manual input or Batch upload, Results |

## 🔧 Algorithm Parameters

### k-Nearest Neighbors
```
- k: 1-20 (recommended: 5)
- Distance: Euclidean/Manhattan/Minkowski
- Weighted: Yes/No (default: No)
```

### Random Forest
```
- Max Depth: 1-20 (recommended: 10)
- Criterion: Gini/Entropy (default: Gini)
- N Estimators: 1-100 (recommended: 50)
- Feature Importance: Yes/No
```

### Naive Bayes
```
- Type: Gaussian/Multinomial/Bernoulli
- Smoothing (alpha): 0.0-1.0 (default: 1.0)
- Calibration: Yes/No
```

## 📊 Performance Metrics

| Metric | Meaning |
|--------|---------|
| Accuracy | Correct predictions / Total predictions |
| Precision | True Positives / (TP + FP) |
| Recall | True Positives / (TP + FN) |
| F1-Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Area under ROC curve (0-1) |

## 🎨 Color Reference

```
Primary Actions:    #2563eb (Blue)
Success States:     #10b981 (Green)
Warnings:          #f59e0b (Orange)
Errors:            #ef4444 (Red)
Background:        #f3f4f6 (Light Gray)
Cards:             #ffffff (White)
```

## 📁 File Locations

```
c:\Users\Ahmad Zain\Desktop\ML Assignment 4\
├── GUI_Preview_Dashboard.ipynb        # All GUI previews
├── requirements.txt                   # Dependencies
├── PROJECT_SETUP.md                   # This documentation
├── QUICK_REFERENCE.md                 # Quick guide (this file)
└── Part*.png                          # 5 GUI preview images
```

## 🚀 Typical Workflow

1. **Open Dataset Page** → Upload CSV/Excel file
2. **View Data** → See preview and statistics
3. **Select Features** → Choose columns to use
4. **Pick Algorithm** → Configure parameters
5. **Train Model** → Set split ratio and train
6. **View Results** → Check accuracy and metrics
7. **Visualize** → View comparison charts
8. **Make Predictions** → Enter data and predict

## 📈 Expected Accuracy Ranges

| Dataset | kNN | Random Forest | Naive Bayes |
|---------|-----|---------------|-------------|
| Iris | 95%+ | 95%+ | 90%+ |
| Breast Cancer | 90%+ | 95%+ | 85%+ |
| Wine | 85%+ | 90%+ | 80%+ |
| Fraud | 80%+ | 85%+ | 75%+ |

## 🔍 Troubleshooting

**Issue**: Import errors
- Solution: Run `pip install -r requirements.txt`

**Issue**: Data preview not showing
- Solution: Make sure CSV/Excel file format is correct

**Issue**: Model not training
- Solution: Check if features and target are properly selected

**Issue**: Visualizations not displaying
- Solution: Verify at least one model is trained

## 📞 Important Notes

- All 5 GUI mockups are ready as reference images
- Virtual environment must be activated before running
- Dataset should not have missing values in key columns
- Maximum dataset size: 100,000 rows (for performance)
- All predictions are real-time (no batch processing delay)

---
**Version**: 1.0  
**Last Updated**: January 21, 2026  
**Status**: ✅ Ready for Development
