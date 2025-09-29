### Healthcare Premium Prediction 🏥 | End-to-End ML Project

This project predicts annual healthcare insurance premiums using a machine learning model. The model analyzes 12 key personal and health-related factors—such as **age, BMI, smoking status, and genetic risk**—to provide an accurate estimate.

A key finding from this project was that a single model was insufficient, performing poorly on the younger demographic (\<25 years). To solve this, the data was segmented by age, and **two separate XGBoost models** were trained. This tailored approach increased the overall prediction accuracy to an impressive **99% R² score**.

The final model is deployed as an interactive web application using **Streamlit**.

### Key Features

  * **High Accuracy:** Achieves a 99% R² score through a strategic age-based segmentation model.
  * **Comprehensive Data:** Utilizes 12 diverse features, including lifestyle choices, medical history, and income level for robust predictions.
  * **Interactive UI:** A user-friendly web app built with Streamlit allows for easy, real-time premium estimation.
  * **End-to-End Workflow:** Covers the full machine learning lifecycle from data cleaning and feature engineering to model deployment.

### Tech Stack

  * **Programming Language:** Python
  * **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
  * **Deployment:** Streamlit, Joblib

### How to Use

1.  Clone the repository:
    ```bash
    git clone https://github.com/ghulvasu/ml-project-health-premium-prediction.git
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
