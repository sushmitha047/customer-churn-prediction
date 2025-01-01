import os
import pandas as pd
import pickle
import numpy as np
import streamlit as st
from openai import OpenAI
import utils as ut

# Initializing the OpenAI client using the GROQ API key from environment variables
client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=st.secrets["GROQ_API_KEY"])


# Function to load machine learning models from pickle files
def load_models(filename):
    """
    Load a machine learning model from a pickle file.

    Args:
        filename (str): Path to the pickle file.

    Returns:
        Object: Loaded machine learning model.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


# Load multiple models from the models directory
xgboost_model = load_models("models/xgb_model.pkl")
random_forest_model = load_models("models/rf_model.pkl")
knn_model = load_models("models/knn_model.pkl")


# Function to prepare input data for prediction
def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_of_products_purchased, has_credit_card, is_active_member,
                  estimated_salary):
    """
    Prepare customer input data for model prediction.

    Args:
        credit_score, location, gender, age, tenure, balance, 
        num_of_products_purchased, has_credit_card, is_active_member, 
        estimated_salary: Customer data fields.

    Returns:
        DataFrame: Prepared input data for prediction.
        dict: Input data as a dictionary.
    """
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products_purchased,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


# Function to make predictions based on customer input data
def make_predictions(input_df, input_dict):
    """
    Predict the probability of customer churn using different models.

    Args:
        input_df (DataFrame): Prepared input data.
        input_dict (dict): Input data as a dictionary.

    Returns:
        float: Average probability of churn.
    """
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'RandomForest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    }
    avg_probability = np.mean(list(probabilities.values()))

    # Display model predictions and probabilities using Streamlit
    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability: .2%} probability of churning."
        )

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability


# Function to explain the prediction based on the probability of churn
def explain_prediction(probability, input_dict, surname):
    """
    Generate an explanation for the predicted churn probability.

    Args:
        probability (float): Probability of the customer churning.
        input_dict (dict): Customer's information used for prediction.
        surname (str): Customer's surname.

    Returns:
        str: A natural language explanation of the prediction.
    """
    prompt = f"""You are an expert data scientist at a bank, where you specialize in 
    interpreting and explaining predictions of machine learning models.
    Your machine learning model has predicted that a customer named {surname} has a 
    {round(probability * 100, 1)}% probability of churning, based on the information 
    provided below.

    Here is the customer's information:
    {input_dict}

    Here are the machine learning model's top 10 most important features for 
    predicting churn:

              Feature | Importance
    ------------------------------------
        NumOfProducts | 0.323888
       IsActiveMember | 0.164146
                  Age | 0.109550
    Geography_Germany | 0.091373
              Balance | 0.052786
     Geography_France | 0.046463
        Gender_Female | 0.045283
      Geography_Spain | 0.036855
          CreditScore | 0.035005
      EstimatedSalary | 0.032655
            HasCrCard | 0.031940
               Tenure | 0.030054
          Gender_Male | 0.000000

    {pd.set_option('display.max_columns', None)}

    Here are summary statistics for churned customers:
    {churn_pd[churn_pd['Exited'] == 1].describe()}

    Here are summary statistics for non-churned customers:
    {churn_pd[churn_pd['Exited'] == 0].describe()}

    - If the customer has over a 40% risk of churning, generate a 3-sentence 
    explanation of why they are at risk of churning.
    - If the customer has less than a 40% risk of churning, generate a 3-sentence 
    explanation of why they might not be at risk of churning.
    - Your explanation should be based on the customer's information, the summary 
    statistics of churned and non-churned customers, and the feature importances 
    provided.

    ADDITIONAL INSTRUCTIONS:
    1. **Do not list all raw feature values** for the customer verbatim (e.g., “CreditScore = X, Age = Y, etc.”). Only highlight the most relevant features (especially from the feature importance list) and compare them to typical churners/non-churners.
    2. **Use the summary statistics accurately** and carefully. Do not fabricate or misstate numeric comparisons. Use phrasing like “above/below the average” or “similar to churned customers,” as appropriate, and ensure the comparisons are consistent with the data.
    3. **Focus on why the model flags high or low churn risk** by mentioning key features and how they differ from churned or non-churned profiles.
    4. **Avoid step-by-step calculation language** (“Let’s calculate…”) and instead provide a concise interpretation of the model’s prediction.
    5. **Use a clear, professional tone** suitable for Global Trust Bank employees. Keep the final explanation to 3 sentences, directly addressing the key drivers behind the predicted churn risk.
    """
    print("EXPLANATION PROMPT:", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    return raw_response.choices[0].message.content


# Function to generate a personalized email based on churn predictions
def generate_email(probability, input_dict, explanation, surname):
    """
    Generate a personalized email encouraging customer retention.

    Args:
        probability (float): Probability of the customer churning.
        input_dict (dict): Customer's information used for prediction.
        explanation (str): Explanation of the churn prediction.
        surname (str): Customer's surname.

    Returns:
        str: A personalized email message for the customer.
    """
    prompt = f"""You are a manager at Global Trust Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.

    You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

    Here is the customer's information:
    {input_dict}

    Here is some explanation as to why the customer might not be at risk of churning: {explanation}

    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

    Make sure to list out a set of incentives to stay based on their information, in bullet point format.

    ADDITIONAL INSTRUCTIONS:
    1. Format the email as a standard professional letter:
       - Include a clear subject line at the top.
       - Begin the email with "Dear [Customer Name]," on its own line.
       - Structure the body into paragraphs or bullet points as appropriate.

    2. For the conclusion and signature:
       - Use a separate closing line like "Warm regards," followed by a line break.
       - On the next line, place the manager’s full name (e.g., “Elise Guerra”).
       - On the following line, place the manager’s title (e.g., “Manager, Customer Loyalty and Experience”).
       - Finally, list “Global Trust Bank” on its own line.

    3. Ensure the email reads in a concise, courteous, and customer-friendly manner, matching the tone of a genuine bank communication.

    4. Do not merge the name, title, and bank name into a single line; each should appear on its own line after the closing.
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    print("\n\nEMAIL PROMPT:", prompt)

    return raw_response.choices[0].message.content


# Main application logic
st.title("Customer Churn Prediction")

# Load customer data
churn_pd = pd.read_csv("churn.csv")
customers = [
    f"{row['CustomerId']} - {row['Surname']}"
    for _, row in churn_pd.iterrows()
]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]
    selected_customer = churn_pd.loc[churn_pd["CustomerId"] ==
                                     selected_customer_id].iloc[0]

    # Collect user input through Streamlit widgets
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer["CreditScore"]))
        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer["Geography"]))
        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer["Gender"] == "Male" else 1)
        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer["Age"]))
        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer["Tenure"]))

    with col2:
        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer["Balance"]))
        num_of_products_purchased = st.number_input(
            "Number of Products Purchased",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"]))
        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer["HasCrCard"]))
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer["IsActiveMember"]))
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"]))

    # Prepare input data and make predictions
    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance,
                                         num_of_products_purchased,
                                         has_credit_card, is_active_member,
                                         estimated_salary)

    avg_probability = make_predictions(input_df, input_dict)

    # --- (Section Separator) ---
    st.markdown("---")

    # Generate explanation and personalized email
    explanation = explain_prediction(avg_probability, input_dict,
                                     selected_customer['Surname'])
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    # --- (Section Separator) ---
    st.markdown("---")

    email = generate_email(avg_probability, input_dict, explanation,
                           selected_customer['Surname'])
    st.subheader("Personalized Email to Customer")
    st.code(email, language=None)
