# ML Classification Models Dashboard

A comprehensive machine learning classification project that implements 6 different classification algorithms and provides an interactive Streamlit web application for model evaluation and comparison.

## 📌 Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models on a chosen dataset and deploy an interactive web application for demonstration. This project covers the complete end-to-end ML deployment workflow: data preprocessing, model training, evaluation, UI design, and cloud deployment.

The application allows users to:
- Upload their own classification datasets
- Select and train different ML models
- View comprehensive evaluation metrics
- Compare model performances visually

## 📊 Dataset Description

**Dataset Name**: Mobile Phone Specifications Dataset

**Source**: Test dataset with 1000 mobile phone records

**Dataset Characteristics**:
| Property | Value |
|----------|-------|
| Number of Instances | 1000 |
| Number of Features | 20 (excluding ID) |
| Number of Binary Features | 10 |
| Number of Numeric Features | 10 |
| Missing Values | 0 |
| Target Variable | Not specified in test set |

**Feature Description**:
| Feature Name | Type | Range | Description |
|--------------|------|-------|-------------|
| battery_power | Numeric (int) | 0-2000 | Battery capacity in mAh |
| blue | Binary (0/1) | 0-1 | Has Bluetooth support |
| clock_speed | Numeric (float) | 0.5-3.0 | Processor clock speed in GHz |
| dual_sim | Binary (0/1) | 0-1 | Dual SIM support |
| fc | Numeric (int) | 0-20 | Front camera megapixels |
| four_g | Binary (0/1) | 0-1 | 4G support |
| int_memory | Numeric (int) | 2-128 | Internal memory in GB |
| m_dep | Numeric (float) | 0.1-1.0 | Mobile depth in cm |
| mobile_wt | Numeric (int) | 80-200 | Mobile weight in grams |
| n_cores | Numeric (int) | 1-8 | Number of processor cores |
| pc | Numeric (int) | 0-20 | Primary camera megapixels |
| px_height | Numeric (int) | 0-1600 | Pixel resolution height |
| px_width | Numeric (int) | 500-2000 | Pixel resolution width |
| ram | Numeric (int) | 256-3900 | RAM in MB |
| sc_h | Numeric (int) | 5-19 | Screen height in cm |
| sc_w | Numeric (int) | 0-16 | Screen width in cm |
| talk_time | Numeric (int) | 2-20 | Talk time in hours |
| three_g | Binary (0/1) | 0-1 | 3G support |
| touch_screen | Binary (0/1) | 0-1 | Touch screen support |
| wifi | Binary (0/1) | 0-1 | WiFi support |

**Data Preprocessing Steps**:
1. No missing values detected in the dataset
2. All features are numeric or binary (no encoding needed)
3. Two float features (clock_speed, m_dep) for certain models requiring scaling
4. Standard scaling applied for distance-based models (KNN, Logistic Regression)
5. Data ready for direct model training without further preprocessing

## 🤖 Models Used

The following 6 classification models were implemented and evaluated:

1. **Logistic Regression** - Linear model for classification
2. **Decision Tree Classifier** - Tree-based interpretable model
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier
5. **Random Forest (Ensemble)** - Ensemble of decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble

### 📈 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| K-Nearest Neighbors | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

> **Note**: Replace the `0.XXXX` values with your actual results after running the models on your chosen dataset.

### 📝 Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | [e.g., "Performs well as a baseline model. Good for linearly separable data. Fast training time and interpretable coefficients."] |
| Decision Tree | [e.g., "Prone to overfitting without proper pruning. Provides interpretable decision rules. Feature importance is easily extractable."] |
| K-Nearest Neighbors | [e.g., "Sensitive to feature scaling and curse of dimensionality. Performance depends on optimal k value. Slow prediction for large datasets."] |
| Naive Bayes | [e.g., "Fast training and prediction. Works well with high-dimensional data. Assumes feature independence which may not hold."] |
| Random Forest (Ensemble) | [e.g., "Reduces overfitting compared to single decision tree. Good handling of imbalanced data. Provides robust feature importance."] |
| XGBoost (Ensemble) | [e.g., "Best performing model with highest accuracy. Handles missing values well. Requires careful hyperparameter tuning."] |

## 🚀 Streamlit App Features

The deployed Streamlit application includes:

1. **📂 Dataset Upload** - Upload CSV files for classification
2. **🔍 Model Selection** - Dropdown to select from 6 different models
3. **📊 Evaluation Metrics Display** - Shows Accuracy, AUC, Precision, Recall, F1, MCC
4. **📈 Confusion Matrix** - Visual confusion matrix heatmap
5. **📋 Classification Report** - Detailed per-class metrics
6. **📊 Model Comparison** - Side-by-side comparison of all models
7. **🌳 Feature Importance** - For tree-based models

## 📁 Project Structure

```
project-folder/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── model/                    # Model files and training scripts
│   └── train_models.py       # Script to train all models
│
└── data/                     # Dataset files (optional)
    ├── data.csv              # Training dataset
    └── test_data.csv         # Test dataset
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 🌐 Deployment

### Streamlit Community Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New App"
5. Select your repository and branch
6. Choose `app.py` as the main file
7. Click "Deploy"

**Live App Link**: [Your Streamlit App URL]

## 📊 How to Use

1. **Upload Dataset**: Click on "Upload CSV file" in the sidebar
   - Ensure your CSV has the target variable as the last column
   - Minimum 12 features and 500 instances recommended

2. **Select Model**: Choose a classification model from the dropdown

3. **View Results**: 
   - Single Model Evaluation tab shows detailed metrics
   - Model Comparison tab compares all 6 models
   - Feature Analysis tab shows feature importance

## 📚 Evaluation Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correct predictions |
| **AUC** | Area Under the ROC Curve - measures discrimination ability |
| **Precision** | Proportion of true positives among predicted positives |
| **Recall** | Proportion of true positives among actual positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **MCC** | Matthews Correlation Coefficient - balanced measure for imbalanced data |

## 🔗 Links

- **GitHub Repository**: [Your GitHub Repo Link]
- **Live Streamlit App**: [Your Streamlit App Link]

## 👨‍💻 Author

**Name**: Ipsita Nanda  
**Program**: M.Tech (AIML/DSE)  
**Institution**: BITS Pilani  
**Course**: Machine Learning - Assignment 2

## 📝 License

This project is for educational purposes as part of the Machine Learning course at BITS Pilani.

---

*Built with ❤️ using Python, Scikit-learn, and Streamlit*
