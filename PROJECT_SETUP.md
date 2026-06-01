# Interactive ML Dashboard - Project Setup Documentation

## 📋 Project Overview
This is an end-to-end Machine Learning Classification Dashboard with GUI, built using Streamlit framework. The application allows users to upload datasets, configure ML algorithms, train models, visualize results, and make predictions.

## ✅ Setup Status: COMPLETE

### Environment Information
- **Python Version**: 3.14.2
- **Virtual Environment**: .venv
- **OS**: Windows
- **Framework**: Streamlit

## 📦 Installed Dependencies

### Core ML Libraries
- **scikit-learn** (v1.3.2) - Machine learning algorithms
- **pandas** (v2.1.3) - Data manipulation and analysis
- **numpy** (v1.24.3) - Numerical computing

### Visualization Libraries
- **matplotlib** (v3.8.2) - Static plots and visualizations
- **plotly** (v5.18.0) - Interactive charts and dashboards
- **pillow** (v10.1.0) - Image processing

### Web Framework & Utilities
- **streamlit** (v1.28.1) - Web application framework
- **openpyxl** (v3.1.2) - Excel file handling
- **scipy** (v1.11.4) - Scientific computing

## 📁 Project Structure

```
ML Assignment 4/
├── GUI_Preview_Dashboard.ipynb          # Jupyter notebook with all GUI previews
├── requirements.txt                      # Project dependencies
├── Part1_Dataset_Management.png          # GUI Preview - Data upload & stats
├── Part2_Algorithm_Configuration.png     # GUI Preview - Algorithm settings
├── Part3_Training_Evaluation.png         # GUI Preview - Model training
├── Part4_Visualization_Dashboard.png     # GUI Preview - Analytics dashboards
└── Part5_Prediction_Interface.png        # GUI Preview - Prediction interface
```

## 🎨 Application Architecture

### Part 1: Dataset Management Module
**Features:**
- Drag & drop file upload (CSV, Excel)
- Data preview (first 10 rows with scrolling)
- Basic statistics display:
  - Mean, Median, Standard Deviation
  - Missing values count
  - Data shape (rows × columns)
- Feature multi-select checkboxes
- Target variable dropdown
- Data type inference

### Part 2: Algorithm Selection & Configuration Panel
**Three ML Algorithms Supported:**

1. **k-Nearest Neighbors (kNN)**
   - k value slider (1-20)
   - Distance metric dropdown (Euclidean, Manhattan, Minkowski)
   - Weighted voting checkbox
   - Cross-validation options

2. **Random Forest**
   - Max depth slider (1-20)
   - Criterion selection (Gini, Entropy)
   - N_estimators slider (1-100)
   - Feature importance display option
   - Balanced class weights

3. **Naive Bayes**
   - Algorithm type dropdown (Gaussian, Multinomial, Bernoulli)
   - Smoothing parameter (alpha) input
   - Probability calibration checkbox
   - Laplace smoothing option

### Part 3: Training & Evaluation Module
**Features:**
- Train/Test split slider (60/40, 70/30, 80/20)
- Cross-validation k-fold selection (3, 5, 10)
- Training progress bar
- Real-time performance metrics:
  - Training Accuracy
  - Testing Accuracy
  - Precision Score
  - Recall Score
  - F1-Score
  - ROC-AUC (for binary classification)
- Confusion Matrix visualization
- Classification Report

### Part 4: Visualization Dashboard
**Three Main Visualizations:**

1. **Algorithm Comparison Chart**
   - Bar chart comparing accuracy across all trained models
   - Interactive tooltips with detailed metrics
   - PNG export functionality
   - Sortable by accuracy

2. **Feature Importance Plot**
   - Horizontal bar chart (top 10 features)
   - Color-coded by importance score
   - Only visible for tree-based models
   - Download as PNG

3. **Decision Boundary Visualization**
   - Scatter plot with decision regions
   - Feature pair selection (any two features)
   - Training vs. misclassified points highlighting
   - Support for 2D feature subsets

### Part 5: Prediction Interface
**Two Input Methods:**

1. **Manual Input**
   - Form with input fields for each feature
   - Real-time validation
   - Clear/Reset button

2. **Batch Upload**
   - CSV file upload for multiple predictions
   - Batch processing with progress tracking
   - Predictions for all rows at once

**Output Features:**
- Predicted class with confidence score
- Probability distribution for all classes
- Detailed confidence percentages
- Export predictions to CSV
- Download results as JSON

## 🎯 Recommended Datasets

### 1. Iris Dataset (Recommended for Start)
- **Source**: Built-in scikit-learn
- **Size**: 150 samples
- **Features**: 4 (Sepal length, Sepal width, Petal length, Petal width)
- **Classes**: 3 (Iris species)
- **Complexity**: Easy
- **Best for**: Testing all algorithms quickly

### 2. Breast Cancer Wisconsin Dataset
- **Source**: scikit-learn or Kaggle
- **Size**: 569 samples
- **Features**: 30 (Medical measurements)
- **Classes**: 2 (Benign, Malignant)
- **Complexity**: Medium
- **Best for**: ROC curves, probability calibration

### 3. Wine Quality Dataset
- **Source**: Kaggle
- **Size**: 6,497 samples
- **Features**: 11 (Chemical properties)
- **Classes**: 10 (Quality ratings 0-9)
- **Complexity**: Medium
- **Best for**: Feature importance analysis

### 4. Credit Card Fraud Detection
- **Source**: Kaggle
- **Size**: 284,807 samples
- **Features**: 30 (Transaction data)
- **Classes**: 2 (Fraud, Legitimate)
- **Complexity**: Hard (Imbalanced)
- **Best for**: Algorithm robustness testing

## 🚀 Getting Started

### Installation
```bash
# Clone or download the project
cd "ML Assignment 4"

# Create virtual environment (if not already done)
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run Streamlit app (when created)
streamlit run app.py
```

## 📊 Color Scheme Reference
- **Primary Blue**: #2563eb (Buttons, headers, active state)
- **Secondary Green**: #10b981 (Success, success metrics)
- **Warning Orange**: #f59e0b (Warnings, alerts)
- **Error Red**: #ef4444 (Errors, failures)
- **Background Light**: #f3f4f6 (Page background)
- **Card White**: #ffffff (Component backgrounds)

## 📋 GUI Preview Files (JPEG)
All 5 GUI mockups have been generated as JPEG images:
1. ✅ Part1_Dataset_Management.png (53 KB)
2. ✅ Part2_Algorithm_Configuration.png (49 KB)
3. ✅ Part3_Training_Evaluation.png (41 KB)
4. ✅ Part4_Visualization_Dashboard.png (47 KB)
5. ✅ Part5_Prediction_Interface.png (43 KB)

## 🛠️ Development Checklist

### Phase 1: Project Foundation
- [ ] Set up Streamlit application structure
- [ ] Create main app.py file
- [ ] Set up sidebar navigation
- [ ] Create session state management

### Phase 2: Dataset Module
- [ ] Implement file upload (CSV, Excel)
- [ ] Add data preview display
- [ ] Create statistics calculation
- [ ] Add feature selection interface

### Phase 3: Algorithm Configuration
- [ ] Implement kNN configuration panel
- [ ] Implement Random Forest configuration
- [ ] Implement Naive Bayes configuration
- [ ] Add parameter validation

### Phase 4: Training Pipeline
- [ ] Connect data preprocessing
- [ ] Implement train/test split
- [ ] Add cross-validation logic
- [ ] Display training progress

### Phase 5: Metrics & Evaluation
- [ ] Calculate accuracy metrics
- [ ] Generate confusion matrix
- [ ] Create classification report
- [ ] Display ROC curve (binary)

### Phase 6: Visualizations
- [ ] Create algorithm comparison chart
- [ ] Create feature importance plot
- [ ] Create decision boundary visualization
- [ ] Add export functionality

### Phase 7: Prediction Interface
- [ ] Create manual input form
- [ ] Create batch upload interface
- [ ] Display predictions with confidence
- [ ] Add CSV export feature

### Phase 8: Deployment & Testing
- [ ] Unit testing
- [ ] Integration testing
- [ ] Performance optimization
- [ ] Docker containerization (optional)

## 📝 Notes
- All required packages have been installed
- Virtual environment is active and ready
- GUI mockups are ready for reference during development
- Color scheme is documented for UI consistency
- Project follows Streamlit best practices

## 📞 Support
For any issues or questions during development:
1. Check the GUI previews for reference
2. Refer to requirements.txt for dependencies
3. Follow the recommended datasets for testing

---
**Setup Completed**: January 21, 2026
**Status**: ✅ All Requirements Installed & Ready
