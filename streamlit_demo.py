"""An example of showing geographic data."""

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel  # Assuming your model is a PipelineModel; adjust as necessary
from pyspark.ml.linalg import Vectors

# Function to initialize Spark Session
@st.experimental_singleton
def get_spark_session(app_name="StreamlitApp"):
    return SparkSession.builder.appName(app_name).getOrCreate()

# Function to load the model (adjust the path as necessary)
@st.experimental_singleton
def load_model(model_path="model/rf_model" ):
    spark = get_spark_session()
    return PipelineModel.load(model_path)

# Placeholder for the encoding logic. Need to replace this with your actual encoding function.
def encode_categorical_features(starting_airport, destination_airport, airline_name):
    # This function should return the vector-encoded forms of the inputs.
    # Example: return Vectors.dense([1.0, 0.0]), Vectors.dense([0.0, 1.0, 0.0]), Vectors.dense([0.0, 1.0])
    # The actual implementation will depend on how you've encoded these features during training.
    return Vectors.dense([0.0]), Vectors.dense([0.0]), Vectors.dense([0.0])

# Function to preprocess input and make a prediction
def predict_price(is_basic_economy, is_refundable, is_non_stop, days_before_flight, day, num_legs, all_same, distance, departure_hour, departure_dow_idx, starting_pop, destination_pop, starting_airport, destination_airport, airline_name):
    spark = get_spark_session()
    model = load_model()
    
    # Encoding categorical features
    starting_airport_encoded, destination_airport_encoded, airline_name_encoded = encode_categorical_features(starting_airport, destination_airport, airline_name)
    
    # Preparing the DataFrame with the correct structure as expected by the model
    input_data = [(is_basic_economy, is_refundable, is_non_stop, days_before_flight, day, starting_airport_encoded, destination_airport_encoded, num_legs, all_same, airline_name_encoded, float(distance), departure_hour, departure_dow_idx, float(starting_pop), float(destination_pop))]
    columns = ["isBasicEconomy", "isRefundable", "isNonStop", "days_before_flight", "day", "startingAirport_encoded", "destinationAirport_encoded", "num_legs", "All_Same", "airline_name_encoded", "distance", "departure_hour", "departure_dow_idx", "starting_pop", "destination_pop"]
    
    input_df = spark.createDataFrame(input_data, schema=columns)
    
    # Make predictions
    predictions = model.transform(input_df)
    
    # Extracting the prediction result
    predicted_price = predictions.select("prediction").collect()[0]["prediction"]
    
    return predicted_price

# Streamlit UI
def main():
    st.title("Flight Price Prediction")
    
    # Collecting user inputs
    is_basic_economy = st.selectbox("Is Basic Economy?", [0, 1])
    is_refundable = st.selectbox("Is Refundable?", [0, 1])
    is_non_stop = st.selectbox("Is Non-Stop?", [0, 1])
    days_before_flight = st.number_input("Days Before Flight", min_value=0)
    day = st.number_input("Day of Month of Flight", min_value=1, max_value=31)
    num_legs = st.number_input("Number of Legs", min_value=1, max_value=4)
    all_same = st.selectbox("Are All Legs on the Same Airline?", [0, 1])
    distance = st.number_input("Distance (miles)", value=0.0)
    departure_hour = st.number_input("Departure Hour", min_value=0, max_value=23)
    departure_dow_idx = st.number_input("Departure Day of Week Index (0=Sunday, 6=Saturday)", value=0.0)
    starting_pop = st.number_input("Starting Airport Population", value=0.0)
    destination_pop = st.number_input("Destination Airport Population", value=0.0)
    starting_airport = st.text_input("Starting Airport")
    destination_airport = st.text_input("Ending Airport")
    airline_name = st.text_input("Airline Name")
    
    if st.button("Predict Fare"):
        prediction = predict
