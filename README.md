## 🎣🌐 Phishing Website Detection 

This Jupyter Notebook provides a comprehensive analysis of various machine learning models for phishing website detection. The dataset used in this analysis contains features extracted from URLs to classify them as phishing or legitimate websites.

### 📚 Data and Libraries 

The following libraries are used in this notebook:
- `matplotlib`, `seaborn`: For data visualization.
- `pandas`, `numpy`: For data manipulation and analysis.
- `sklearn`: For machine learning models and evaluation metrics.
- `xgboost`, `tensorflow`: For advanced machine learning models.

The dataset used is loaded from a CSV file named `dataset_phishing.csv`. It contains features extracted from URLs along with the target variable indicating whether the website is phishing or legitimate.

### 📊🔍 Exploratory Data Analysis (EDA) 

- Summary statistics and information about the dataset are provided.
- Distribution of the target variable and its relationship with other features are visualized.

### 🛠️ Preprocessing 

- Label encoding is applied to convert categorical target variable to binary format.
- Data is normalized using standard scaling.
- The dataset is split into training and testing sets.

### 🤖📊 Machine Learning Models 

The following machine learning models are trained and evaluated:
1. Support Vector Machine (SVM)
2. k-Nearest Neighbors (k-NN)
3. Random Forest
4. XGBoost
5. Artificial Neural Network (ANN)

### 🧾📈 Model Evaluation 

- For each model, hyperparameters are tuned using GridSearchCV.
- Performance metrics such as accuracy, precision, recall, and F1-score are calculated.
- Confusion matrices are plotted to visualize the performance of the models.

### ✨ Conclusion 

- The notebook concludes with insights into the best-performing models and suggestions for further improvement.
- Training and validation accuracy/loss plots for the ANN model are provided.
- A detailed confusion matrix is displayed to understand the model's performance in classifying phishing and legitimate websites.

📝 **Author**: Mahmoud Khalil 
