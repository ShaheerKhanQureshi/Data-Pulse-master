import os
import pandas as pd
import streamlit as st
from pycaret.classification import setup, compare_models, pull, save_model
import streamlit as st
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_df(df):
    columns_to_drop = []
    df = df.drop(columns=columns_to_drop)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    st.write("DataFrame after conversion:", df.head())

    original_row_count = df.shape[0]
    df.dropna(inplace=True)
    rows_dropped = original_row_count - df.shape[0]

    st.write("DataFrame after dropping NA:", df.head())

    return df


if os.path.exists('./dataset.csv'):
    df = pd.read_csv('./dataset.csv')


with st.sidebar:
    st.image("logo.jpg")
    st.title("Data Pulse")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Modelling", "Visualization", "Download"])
    st.info("This project application helps you build and explore your data.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)


if choice == "Profiling" and not df.empty:
    st.title("Exploratory Data Analysis")
    profile_df = ProfileReport(df, explorative=True)
    st_profile_report(profile_df)
# else:
#     st.warning("No data available. Please upload a dataset.")


if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        st.write("DataFrame shape before preprocessing:", df.shape)

        df = prepare_df(df)

        st.write("DataFrame shape after preprocessing:", df.shape)

        if df.empty:
            st.error(
                "The dataframe is empty after preprocessing. Please check your data and preprocessing steps.")
        else:
            setup(df, target=chosen_target, verbose=False)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')


if choice == "Visualization":
    st.title("Data Visualization")

    if not df.empty:
        plot_type = st.selectbox("Select Plot Type", [
                                 "Histogram", "Scatter Plot", "Box Plot"])

        # For Histogram
        if plot_type == "Histogram":
            selected_column = st.selectbox(
                "Select Column for Histogram", df.columns)
            num_bins = st.slider("Select Number of Bins", 5, 50, 10)
            plt.figure(figsize=(10, 6))
            sns.histplot(df[selected_column], bins=num_bins, kde=True)
            st.pyplot(plt)

        # For Scatter Plot
        elif plot_type == "Scatter Plot":
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_axis, y=y_axis)
            st.pyplot(plt)

        # For Box Plot
        elif plot_type == "Box Plot":
            selected_column = st.selectbox(
                "Select Column for Box Plot", df.columns)
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[selected_column])
            st.pyplot(plt)
    else:
        st.warning(
            "No data available for visualization. Please upload a dataset.")

if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")