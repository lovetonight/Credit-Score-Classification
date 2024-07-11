import streamlit as st
import json
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)
other_dir = os.path.join(current_dir, "..", "new-model")
other_dir = os.path.abspath(other_dir)

parameter_path = os.path.join(other_dir, "result", "GA.json")
data_path = os.path.join(other_dir, "data", "Lending-Data-Ethereum_Normalized.csv")


with open(parameter_path, "r") as f:
    parameter = np.array(json.load(f))

df = pd.read_csv(data_path)

st.title("Wallet Credit Score")
chain = st.selectbox("Chain", ["Ethereum", "Binance Smart Chain"])
address = st.text_input("Address", "")


# Button to get wallets with the same owners
if st.button("Get Credit Score"):
    if address:
        # st.write(f"Caculating credit score for {address} on {chain}...")
        # Add your logic to fetch wallets here
        result = df[df["address"] == address]
        if len(result) > 0:
            values = np.array(result.drop(columns=["address"]).values.flatten())
            score = int(np.dot(parameter, values))
            label = None
            if 300 <= score < 580:
                label = "Poor"
            elif 580 <= score < 670:
                label = "Fair"
            elif 670 <= score < 740:
                label = "Good"
            elif 740 <= score < 800:
                label = "Very Good"
            elif 800 <= score <= 850:
                label = "Excellent"
            # HTML code to enlarge the text
            html_string = f"""
    Credit score for <strong style='font-size:24px;'>{address}</strong> is <strong style='font-size:24px;'>{score}</strong> belongs to the <strong style='font-size:24px;'>{label}</strong> level
"""

            st.markdown(html_string, unsafe_allow_html=True)
        else:
            st.write(f"No information about this wallet.")
    else:
        st.write("Please enter a valid address.")

# Footer
st.markdown(
    """
    <footer style='text-align: center;'>
        <hr>
        <p>No Copyright ©.</p>
    </footer>
""",
    unsafe_allow_html=True,
)
