import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# Load your trained model (adjust as necessary to match your setup)
@st.cache(allow_output_mutation=True)
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("streamlit_xgboost_model.json")
    return model

model = load_model()

def main():
    st.title("Flight Prediction Dashboard")

    airports = {'ATL': 0, 'BOS': 1, 'CLT': 2, 'DEN': 3, 'DFW': 4, 'DTW': 5,
                'EWR': 6, 'IAD': 7, 'JFK': 8, 'LAX': 9, 'LGA': 10, 'MIA': 11,
                'OAK': 12, 'ORD': 13, 'PHL': 14, 'SFO': 15}
    
    airlines = {'Alaska Airlines': 0, 'American Airlines': 1, 'Boutique Air': 2, 'Cape Air': 3,
                'Contour Airlines': 4, 'Delta': 5, 'Frontier Airlines': 6, 'JetBlue Airways': 7,
                'Key Lime Air': 8, 'Southern Airways Express': 9, 'Spirit Airlines': 10, 
                'Sun Country Airlines': 11, 'United': 12}

    # Define the dropdown options
    days_before_flight_options = list(range(31))
    day_options = list(range(1, 32))
    num_legs_options = list(range(1, 5))
    # distance_options = np.arange(0, 10001, 500).tolist()
    departure_hour_options = list(range(24))
    departure_dow_options = list(range(7))

    # User input fields
    starting_airport = st.selectbox("Starting Airport", list(airports.keys()))
    destination_airport = st.selectbox("Destination Airport", list(airports.keys()))
    airline_name = st.selectbox("Airline Name", list(airlines.keys()))
    is_basic_economy = st.selectbox("Is Basic Economy?", [0, 1])
    is_refundable = st.selectbox("Is Refundable?", [0, 1])
    is_non_stop = st.selectbox("Is Non-Stop?", [0, 1])
    days_before_flight = st.selectbox("Days Before Flight", days_before_flight_options)
    day = st.selectbox("Day of Month of Flight", day_options)
    num_legs = st.selectbox("Number of Legs", num_legs_options)
    all_same = st.selectbox("Are All Legs on the Same Airline?", [0, 1])
    departure_hour = st.selectbox("Departure Hour", departure_hour_options)
    departure_dow_idx = st.selectbox("Departure Day of Week Index (0=Sunday, 6=Saturday)", departure_dow_options)

    if st.button('Predict'):
        # Encode the user input using the dictionaries
        encoded_starting_airport = airports[starting_airport]
        encoded_destination_airport = airports[destination_airport]
        encoded_airline_name = airlines[airline_name]

        # Create a DataFrame for input features
        features = pd.DataFrame({

            'isBasicEconomy': [is_basic_economy],
            'isRefundable': [is_refundable],
            'isNonStop': [is_non_stop],
            'days_before_flight': [days_before_flight],
            'day': [day],
            'num_legs': [num_legs],
            'all_same': [all_same],
            'departure_hour': [departure_hour],
            'departure_dow_idx': [departure_dow_idx],
            'startingAirport_encoded': [encoded_starting_airport],
            'destinationAirport_encoded': [encoded_destination_airport],
            'segmentsAirlineName_encoded': [encoded_airline_name]
        })

        # Make prediction
        prediction = model.predict(features)
        st.write(f"Predicted Total Fare: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
