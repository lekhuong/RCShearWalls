# -*- coding: utf-8 -*-
"""
Created on January 20 2023

@author: Khuong LE NGUYEN / University of Canberra
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd


@st.experimental_memo
# function to convert to superscript
def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)


# function to convert to subscript
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans("".join(normal), "".join(sub_s))
    return x.translate(res)


# loading the saved models

# GBRK = pickle.load(open('GBR.sav', 'rb'))
XGBoost_13Var = pickle.load(open("XGBoost_13Var.sav", "rb"))
XGBoost_10Var = pickle.load(open("XGBoost_10Var.sav", "rb"))
XGBoost_8Var = pickle.load(open("XGBoost_8Var.sav", "rb"))
XGBoost_6Var = pickle.load(open("XGBoost_6Var.sav", "rb"))

# sidebar for navigation
# Icons: https://icons.getbootstrap.com/

with st.sidebar:
    selected = option_menu(
        "Squat Wall - Shear Strength Prediction",
        [
            "Project Description",
            "13 Inputs features",
            "10 Inputs features",
            "08 Inputs features",
            "06 Inputs features",
            "SHAP Values",
        ],
        icons=[
            "server",
            "activity",
            "activity",
            "activity",
            "activity",
            "palette-fill",
        ],
        default_index=0,
    )
    st.write(
        "Contact: K.Le-Nguyen \n\nUniversity of Transport Technology, Vietnam \n\nkhuongln@utt.edu.vn"
    )
    # st.sidebar.markdown('<a href="mailto:khuong.lenguyen@canberra.edu.au">Contact us!</a>', unsafe_allow_html=True)

# Project Description
if selected == "Project Description":
    # page title
    st.title("Project Description")
    st.image(
        "Wall_Description.png",
        caption="Geometry Properties",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    st.write(
        """This project presents a machine-learning approach for predicting the shear strength of Squat RC Walls using data from 639 samples. Ten different machine learning models were trained and tested, including Linear Regression, Support Vector Machine, KNN, Artificial Neural Network, Decision Tree, Random Forest, Gradient Boosting Regression, AdaBoost, CatBoost, and XGBoost. Four of these models were further optimized using Bayesian Optimization. The XGBoost model was found to have the highest predictability with an R2 value greater than 0.96 on the testing data set. The machine learning model's predictability and feature importance were analyzed using the SHAP Value method. The research also proposed and investigated three subdatabases with different numbers of input features using Monte Carlo simulations. The results of the machine learning model were compared to mechanical models based on current standards and demonstrated the high predictability and reliability of the machine learning approach. The best models were subsequently implemented online for practical application. It is important to note that in this application, there are 4 models to predict the shear strength of RC walls using data with 13, 10, 8 and 6 input features.
    """
    )
    st.markdown(
        "**However, it is important to use realistic values for inputs, as using unrealistic values may result in poor predictions. Users are therefore suggested to use the realistic component of input features.**"
    )
    st.subheader("Research Flowchart")
    st.image(
        "Process_Description.png",
        caption="Research Flowchart",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    st.subheader("Data Decription")
    st.markdown(
        "There are four input features categories, namely, geometric dimensions, reinforcement ar-rangements, material properties, and applied axial load. The detailed input features are the height _hw_, length _lw_, web thickness _tw_, flange length _bf_, flange thickness _tf_, concrete compres-sive strength _fck_, vertical web reinforcement ratio _ρv_ and strength _fyv_, horizontal web rein-forcement ratio _ρh_ and strength _fyh_, longitudinal reinforcement ratio _ρL_ and strength _fyL_, and, finally, the applied axial load _P_. The output is simply the shear strength _Vn_."
    )

    pd.options.display.float_format = "{:,.2f}".format
    df = pd.read_csv("datawall639sua1912.csv")
    df = df.applymap("{0:.2f}".format)
    # page title
    st.title("Original Database ")
    st.dataframe(df)

# Prediction Page with 13 inputs
if selected == "13 Inputs features":

    # page title
    st.title("Data with 13 input features")

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        height = st.slider(
            "Height: hw(mm)",
            min_value=150,
            max_value=3100,
            value=800,
        )

    with col2:
        # Glucose = st.text_input('Glucose Level')
        length = st.slider(
            "Length: lw(mm)",
            min_value=400,
            max_value=4000,
            value=1600,
        )

    with col1:
        # SkinThickness = st.text_input('Skin Thickness value')
        web_thickness = st.slider(
            "Web thickness: tw(mm)",
            min_value=10,
            max_value=360,
            value=60,
        )

    with col2:
        flange_thickness = st.slider(
            "Flange thickness: tf(mm)",
            min_value=10,
            max_value=400,
            value=160,
        )

    with col1:
        flange_length = st.slider(
            "Flange length: bf(mm)",
            min_value=30,
            max_value=3100,
            value=60,
        )

    with col2:
        Cconcrete_compressive_strength = st.slider(
            "Concrete compressive strength: fck(MPa)",
            min_value=10,
            max_value=110,
            value=26,
        )

    with col1:
        rhoV = st.slider(
            "Vertical web reinforcement ratio: ρv(%)",
            min_value=0.0,
            max_value=4.0,
            value=0.8,
        )

    with col2:
        fyv = st.slider(
            "Yield strength of the vertical reinforced: fyv(MPa)",
            min_value=0,
            max_value=700,
            value=433,
        )

    with col1:
        rhoH = st.slider(
            "Horizontal web reinforcement ratio: ρh(%)",
            min_value=0.0,
            max_value=4.0,
            value=0.82,
        )

    with col2:
        fyH = st.slider(
            "Yield strength of the horizontal reinforced: fyh(MPa)",
            min_value=0,
            max_value=700,
            value=433,
        )
    with col1:
        rhoL = st.slider(
            "Reinforcement ratios of the flanged element: ρL(%)",
            min_value=0.5,
            max_value=11.0,
            value=4.4,
        )

    with col2:
        fyL = st.slider(
            "Yield strength of the flanged reinforced: fyL(MPa)",
            min_value=200,
            max_value=700,
            value=346,
        )
    with col1:
        Pkk = st.slider(
            "Axial load: P(kN)",
            min_value=0,
            max_value=2400,
            value=150,
        )
    # code for Prediction
    # diab_diagnosis = ''
    ccstrength = ""

    # creating a button for Prediction
    st.subheader("Shear Strength with 13 input features")

    ccsXGB = XGBoost_13Var.predict(
        np.asmatrix(
            [
                height,
                length,
                web_thickness,
                Cconcrete_compressive_strength,
                rhoV / 100.0,
                fyv,
                rhoH / 100.0,
                fyH,
                rhoL / 100.0,
                fyL,
                Pkk,
                flange_thickness,
                flange_length,
            ]
        )
    )

    str1 = "{} kN \n\n Please note that for accurate predictions, it is crucial to use realistic input values.".format(
        np.round(ccsXGB, 2)
    )

    if st.button("Prediction by XGBoost Model"):
        st.success(str1)

# Prediction Page with 10 inputs
if selected == "10 Inputs features":

    # page title
    st.title("Data with 10 input features")

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        height = st.slider(
            "Height: hw(mm)",
            min_value=150,
            max_value=3100,
            value=800,
        )

    with col2:
        length = st.slider(
            "Length: lw(mm)",
            min_value=400,
            max_value=4000,
            value=1600,
        )

    with col1:
        web_thickness = st.slider(
            "Web thickness: tw(mm)",
            min_value=10,
            max_value=360,
            value=60,
        )

    with col2:
        flange_thickness = st.slider(
            "Flange thickness: tf(mm)",
            min_value=10,
            max_value=400,
            value=160,
        )

    with col1:
        flange_length = st.slider(
            "Flange length: bf(mm)",
            min_value=30,
            max_value=3100,
            value=60,
        )

    with col2:
        Concrete_compressive_strength = st.slider(
            "Concrete compressive strength: fck(MPa)",
            min_value=10,
            max_value=110,
            value=26,
        )

    with col1:
        rhoV = st.slider(
            "Vertical web reinforcement ratio: ρv(%)",
            min_value=0.0,
            max_value=4.0,
            value=1.0,
        )

    with col2:
        fyH = st.slider(
            "Yield strength of the horizontal reinforced: fyh(MPa)",
            min_value=0,
            max_value=700,
            value=430,
        )
    with col1:
        rhoL = st.slider(
            "Reinforcement ratios of the flanged element: ρL(%)",
            min_value=0.5,
            max_value=11.0,
            value=4.0,
        )
    with col2:
        Pkk = st.slider(
            "Axial load: P(kN)",
            min_value=0,
            max_value=2400,
            value=0,
        )
    # code for Prediction
    # diab_diagnosis = ''
    ccstrength = ""

    # creating a button for Prediction
    st.subheader("Shear Strength with 10 input features")

    ccsXGB = XGBoost_10Var.predict(
        np.asmatrix(
            [
                height,
                length,
                web_thickness,
                Concrete_compressive_strength,
                rhoV / 100.0,
                fyH,
                rhoL / 100.0,
                Pkk,
                flange_thickness,
                flange_length,
            ]
        )
    )

    str1 = "{} kN \n\n Please note that for accurate predictions, it is crucial to use realistic input values.".format(
        np.round(ccsXGB, 2)
    )

    if st.button("Prediction by XGBoost Model"):
        st.success(str1)

# Prediction Page with 8 inputs
if selected == "08 Inputs features":

    # page title
    st.title("Data with 08 input features")

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        height = st.slider(
            "Height: hw(mm)",
            min_value=150,
            max_value=3100,
            value=800,
        )

    with col2:
        length = st.slider(
            "Length: lw(mm)",
            min_value=400,
            max_value=4000,
            value=1600,
        )

    with col1:
        web_thickness = st.slider(
            "Web thickness: tw(mm)",
            min_value=10,
            max_value=360,
            value=60,
        )

    with col2:
        flange_thickness = st.slider(
            "Flange thickness: tf(mm)",
            min_value=10,
            max_value=400,
            value=160,
        )

    with col1:
        flange_length = st.slider(
            "Flange length: bf(mm)",
            min_value=30,
            max_value=3100,
            value=60,
        )
    with col2:
        Concrete_compressive_strength = st.slider(
            "Concrete compressive strength: fck(MPa)",
            min_value=10,
            max_value=110,
            value=26,
        )
    with col1:
        rhoL = st.slider(
            "Reinforcement ratios of the flanged element: ρL(%)",
            min_value=0.5,
            max_value=11.0,
            value=4.0,
        )
    with col2:
        Pkk = st.slider(
            "Axial load: P(kN)",
            min_value=0,
            max_value=2400,
            value=100,
        )
    # code for Prediction
    # diab_diagnosis = ''
    ccstrength = ""

    # creating a button for Prediction
    st.subheader("Shear Strength with 8 input features")

    ccsXGB = XGBoost_8Var.predict(
        np.asmatrix(
            [
                height,
                length,
                web_thickness,
                Concrete_compressive_strength,
                rhoL / 100.0,
                Pkk,
                flange_thickness,
                flange_length,
            ]
        )
    )

    str1 = "{} kN \n\n Please note that for accurate predictions, it is crucial to use realistic input values.".format(
        np.round(ccsXGB, 2)
    )

    if st.button("Prediction by XGBoost Model"):
        st.success(str1)

# Prediction Page with 6 inputs
if selected == "06 Inputs features":

    # page title
    st.title("Data with 06 input features")

    # getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        height = st.slider(
            "Height: hw(mm)",
            min_value=150,
            max_value=3100,
            value=800,
        )

    with col2:
        length = st.slider(
            "Length: lw(mm)",
            min_value=400,
            max_value=4000,
            value=1600,
        )

    with col1:
        web_thickness = st.slider(
            "Web thickness: tw(mm)",
            min_value=10,
            max_value=360,
            value=60,
        )
    with col2:
        flange_length = st.slider(
            "Flange length: bf(mm)",
            min_value=30,
            max_value=3100,
            value=60,
        )
    with col1:
        rhoL = st.slider(
            "Reinforcement ratios of the flanged element: ρL(%)",
            min_value=0.5,
            max_value=11.0,
            value=4.0,
        )
    with col2:
        Pkk = st.slider(
            "Axial load: P(kN)",
            min_value=0,
            max_value=2400,
            value=100,
        )
    # code for Prediction
    # diab_diagnosis = ''
    ccstrength = ""

    # creating a button for Prediction
    st.subheader("Shear Strength with 6 input features")

    ccsXGB = XGBoost_6Var.predict(
        np.asmatrix(
            [
                height,
                length,
                web_thickness,
                rhoL / 100.0,
                Pkk,
                flange_length,
            ]
        )
    )

    str1 = "{} kN \n\n Please note that for accurate predictions, it is crucial to use realistic input values.".format(
        np.round(ccsXGB, 2)
    )

    if st.button("Prediction by XGBoost Model"):
        st.success(str1)

# SHAP Values
if selected == "SHAP Values":

    # page title
    st.title("SHAP Values")
    st.image(
        "SHAP XGboost - 13Var - All Impact.png",
        caption="SHAP Values",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    st.image(
        "SHAP XGboost-13Var - Average Impact.png",
        caption="Feature Importance",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )


# Data statistic
if selected == "Original Data":
    pd.options.display.float_format = "{:,.2f}".format
    df = pd.read_csv("concrete_for_SHAP.csv")
    df = df.applymap("{0:.2f}".format)
    # page title
    st.title("Original Database ")
    st.dataframe(df)
    st.image(
        "DataVisu.png",
        caption="Hex contour chart of input variables",
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
