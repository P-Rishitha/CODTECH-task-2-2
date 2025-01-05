Name: PODUGU RISHITHA

Company: CODTECH IT SOLUTIONS

ID: CT08DFJ

Domain: Machine Learning

Duration: December 2024 to January 2025

Mentor: NEELA SANTHOSH KUMAR

---

## **Sentiment Analysis Using Logistic Regression**

This repository provides a Python implementation of sentiment analysis on customer reviews using **TF-IDF vectorization** and a **logistic regression model**. The dataset is small and generated directly in the code, making it ideal for learning or quick demonstrations.

---

**Output**

![Screenshot 2025-01-05 12 49 10](https://github.com/user-attachments/assets/a3e343a4-18b2-4a63-a906-f6072b2b0cb2)

---

### **Overview**

The code demonstrates:
1. Creation of a small dataset of customer reviews with sentiments (positive or negative).
2. Text feature extraction using TF-IDF vectorization.
3. Splitting the dataset into training and testing sets.
4. Training a logistic regression model for binary sentiment classification.
5. Evaluating the model's performance using a classification report and accuracy metric.

---

### **Steps in the Code**

1. **Dataset Creation**  
   A small dataset of 10 customer reviews is created directly in the script. Each review is labeled with a sentiment:
   - `1`: Positive sentiment
   - `0`: Negative sentiment

2. **TF-IDF Vectorization**  
   The reviews are transformed into numerical representations using **TF-IDF (Term Frequency-Inverse Document Frequency)** to prepare the data for the machine learning model.

3. **Data Splitting**  
   The dataset is divided into training and testing sets using an 80/20 split with `train_test_split`.

4. **Model Training**  
   A **logistic regression model** is trained on the TF-IDF features and their corresponding sentiment labels.

5. **Model Evaluation**  
   Predictions are made on the test set, and the model's performance is evaluated using:
   - **Classification Report:** Provides precision, recall, F1-score, and support for each class.
   - **Accuracy Score:** Measures the overall accuracy of the model.

---

### **Requirements**

- Python 3.7 or higher
- Libraries: pandas, scikit-learn

Install the dependencies using:
```bash
pip install pandas scikit-learn
```

---

### **Usage**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-logistic-regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sentiment-analysis-logistic-regression
   ```
3. Run the script:
   ```bash
   python sentiment_analysis.py
   ```
   
---

### **Extensions**

You can extend the project by:
- Using a larger and more diverse dataset.
- Applying advanced models like Support Vector Machines (SVM) or Neural Networks.
- Adding data preprocessing steps like stemming, lemmatization, or stopword removal.
- Visualizing performance metrics for better interpretability.

---

### **Contributing**

Contributions are welcome! Feel free to:
- Improve the model's performance.
- Add new features, such as hyperparameter tuning or cross-validation.
- Extend the dataset with more customer reviews.
