# gym-customer-churn-prediction
This project implements a Machine Learning model to predict whether a gym customer will churn (cancel their membership) or not, using Logistic Regression. The model was selected after comparing two binary classification algorithms—Logistic Regression and Decision Tree—and evaluating their performance using Confusion Matrix and ROC Curve analysis. The results demonstrated that Logistic Regression achieved higher accuracy and better generalization, making it the optimal choice for this prediction task.

Key Features
Binary Classification Model: Predicts customer churn (1) or retention (0).

Algorithm Comparison: Evaluated Logistic Regression vs. Decision Tree, confirming Logistic Regression's superior performance.

Model Evaluation: Assessed using Confusion Matrix (precision, recall, accuracy) and ROC-AUC Curve to ensure reliability.

Deployment: Built a Flask-based API to serve predictions in real-time.

Tech Stack: Python, Scikit-Learn (for model training), Pandas (data processing), and Joblib (model serialization).

Why Logistic Regression?
High Accuracy: Demonstrated near-optimal performance on the dataset.

Interpretability: Provides clear probabilistic outputs and feature importance.

Efficiency: Lightweight and fast for production deployment.

This solution helps gym businesses proactively retain customers by identifying at-risk members and enabling targeted retention strategies.

Repository Structure:

/model/ – Trained Logistic Regression model (serialized with Joblib).

/data/ – Preprocessed dataset for training/testing.

/app/ – Flask API routes for prediction endpoints.

requirements.txt – Dependencies for easy setup.

How to Use:

Clone the repository.

Install dependencies (pip install -r requirements.txt).

Run the Flask app (python app.py).

Send POST requests with customer data to the API for churn predictions.

Future Improvements:

Integrate more features (e.g., customer engagement metrics).

Deploy on cloud platforms (AWS/GCP) for scalability.

This project serves as a scalable template for binary classification tasks in customer analytics. Contributions and feedback are welcome!
