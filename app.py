import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image



pickle_in = open("final_dtmodel.pkl", "rb")
classifier = pickle.load(pickle_in)



def welcome():
    return "Welcome All"



def customer_segmentation(balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
             cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure):
    """Let's predict in which group customers belong
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: balance
        in: query
        type: number
        required: true
      - name: balance_frequency
        in: query
        type: number
        required: true
      - name: purchases
        in: query
        type: number
        required: true
      - name: oneoff_purchases
        in: query
        type: number
        required: true
      - name: installments_purchases
        in: query
        type: number
        required: true
      - name: cash_advance
        in: query
        type: number
        required: true
      - name: purchases_frequency
        in: query
        type: number
        required: true
      - name: oneoff_purchases_frequency
        in: query
        type: number
        required: true
      - name: purchases_installment_frequency
        in: query
        type: number
        required: true
      - name: cash_advance_frequency
        in: query
        type: number
        required: true
      - name: cash_advance_trx
        in: query
        type: number
        required: true
      - name: purchases_trx
        in: query
        type: number
        required: true
      - name: credit_limit
        in: query
        type: number
        required: true
      - name: payments
        in: query
        type: number
        required: true
      - name: minimum_payments
        in: query
        type: number
        required: true
      - name: prc_full_payment
        in: query
        type: number
        required: true
      - name: tenure
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
             cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]

    prediction = classifier.predict(data)
    print(prediction)
    return prediction


def main():
    st.title("Customer Segmentation")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Customer Segmentation ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    balance = st.text_input("balance", "Type Here")
    balance_frequency = st.text_input("balance_frequency", "Type Here")
    purchases = st.text_input("purchases", "Type Here")
    oneoff_purchases = st.text_input("oneoff_purchases", "Type Here")
    installments_purchases = st.text_input("installments_purchases", "Type Here")
    cash_advance = st.text_input("cash_advance", "Type Here")
    purchases_frequency = st.text_input("purchases_frequency", "Type Here")
    oneoff_purchases_frequency = st.text_input("oneoff_purchases_frequency", "Type Here")
    purchases_installment_frequency = st.text_input("purchases_installment_frequency", "Type Here")
    cash_advance_frequency = st.text_input("cash_advance_frequency", "Type Here")
    cash_advance_trx = st.text_input("cash_advance_trx", "Type Here")
    purchases_trx = st.text_input("purchases_trx", "Type Here")
    credit_limit = st.text_input("credit_limit", "Type Here")
    payments = st.text_input("payments", "Type Here")
    minimum_payments = st.text_input("minimum_payments", "Type Here")
    prc_full_payment = st.text_input("prc_full_payment", "Type Here")
    tenure = st.text_input("tenure", "Type Here")
    result = ""
    if st.button("Predict"):
        result = customer_segmentation(balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
             purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
             cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure)
    st.success('The output is {}'.format(result))
    if st.button("About"):
                st.text("Built with Streamlit")


if __name__ == '__main__':
    main()