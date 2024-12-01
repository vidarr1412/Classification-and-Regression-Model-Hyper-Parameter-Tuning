
import streamlit as st
import io
import numpy as np
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from datetime import datetime
import joblib


#Purpose to be able to display dataframe
@st.cache_resource
def load_train_models():
# Load dataset
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        st.markdown("<h1 style='text-align: center;'>Sea Level Regression Model Tester and Trainer</h1>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.write(data.head())

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]
            

            accuracies=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, train_size=0.8)

            #LR
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Linear Regression", "MAE": f"{mae:.2f}"})
            #gbm
            ada_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
            ada_regressor.fit(X_train, y_train)
            y_pred = ada_regressor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Adaboost", "MAE": f"{mae:.2f}"})
            #DT
            lasso_regressor = Lasso(alpha=0.1)  # You can tune the alpha parameter
            lasso_regressor.fit(X_train, y_train)
            y_pred = lasso_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Lasso Regression", "MAE": f"{mae:.2f}"})

            # Ridge Regression model
            ridge_regressor = Ridge(alpha=0.1)  # You can tune the alpha parameter
            ridge_regressor.fit(X_train, y_train)
            y_pred = ridge_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Ridge Regression", "MAE": f"{mae:.2f}"})

            #mlp
            mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)  # Adjust hidden layers and max_iter
            mlp_regressor.fit(X_train, y_train)
            y_pred = mlp_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "MLP Regressor", "MAE": f"{mae:.2f}"})
            #rft
            rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune the number of trees (n_estimators)
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Random Forest Regressor", "MAE": f"{mae:.2f}"})

            #svm
            svm_regressor = SVR(kernel='poly', C=10, epsilon=0.1)  # Adjust C, epsilon, and kernel type for tuning
            svm_regressor.fit(X_train, y_train)
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "SVM Regressor", "MAE": f"{mae:.2f}"})

            #elastic
            elastic_net_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust alpha and l1_ratio parameters
            elastic_net_regressor.fit(X_train, y_train)
            y_pred = elastic_net_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Elastic Net", "MAE": f"{mae:.2f}"})

            #dt
            decision_tree_regressor = DecisionTreeRegressor(random_state=42)  # You can tune max_depth, min_samples_split, etc.
            decision_tree_regressor.fit(X_train, y_train)
            y_pred = decision_tree_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Decision Tree", "MAE": f"{mae:.2f}"})

            #knn
            knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can tune the number of neighbors
            knn_regressor.fit(X_train, y_train)
            y_pred = knn_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "KNN Regressor", "MAE": f"{mae:.2f}"})
    
            return accuracies

#Purpose disable display dataframe from load_train_models2
@st.cache_resource
def load_train_models2():
# Load dataset
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            accuracies2=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, train_size=0.8)

        # LR
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
          
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Linear Regression", "MAE": f"{mae:.2f}"})
        #GBM
            ada_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
            ada_regressor.fit(X_train, y_train)
            y_pred = ada_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Adaboost", "MAE": f"{mae:.2f}"})
        #DT
            lasso_regressor = Lasso(alpha=0.1)  # You can tune the alpha parameter
            lasso_regressor.fit(X_train, y_train)
            y_pred = lasso_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Lasso Regression", "MAE": f"{mae:.2f}"})

        # Ridge Regression model
            ridge_regressor = Ridge(alpha=0.1)  # You can tune the alpha parameter
            ridge_regressor.fit(X_train, y_train)
            y_pred = ridge_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Ridge Regression", "MAE": f"{mae:.2f}"})
        #mlp
            mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)  # Adjust hidden layers and max_iter
            mlp_regressor.fit(X_train, y_train)
            y_pred = mlp_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "MLP Regressor", "MAE": f"{mae:.2f}"})
        #rft
            rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune the number of trees (n_estimators)
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Random Forest Regressor", "MAE": f"{mae:.2f}"})

        # Support Vector Machine (SVM) Regressor model
            svm_regressor = SVR(kernel='poly', C=10, epsilon=0.1) # Adjust C, epsilon, and kernel type for tuning
            svm_regressor.fit(X_train, y_train)
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "SVM Regressor", "MAE": f"{mae:.2f}"})
        #elastic
            elastic_net_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust alpha and l1_ratio parameters
            elastic_net_regressor.fit(X_train, y_train)
            y_pred = elastic_net_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Elastic Net", "MAE": f"{mae:.2f}"})
        #dt
            decision_tree_regressor = DecisionTreeRegressor(random_state=42)  # You can tune max_depth, min_samples_split, etc.
            decision_tree_regressor.fit(X_train, y_train)
            y_pred = decision_tree_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Decision Tree", "MAE": f"{mae:.2f}"})
        #knn   
            knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can tune the number of neighbors
            knn_regressor.fit(X_train, y_train)
            y_pred = knn_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "KNN Regressor", "MAE": f"{mae:.2f}"})
            
            return accuracies2
    
#stage1  
#display Dataframe and Table of MAE Results

def stage1():
    
    accuracies2 = load_train_models2()
    st.sidebar.title("NAVIGATION")

    if st.sidebar.button("Model Table Summary"):
        
        st.rerun() 
    elif st.sidebar.button("Model Comparison Graph"):
        st.session_state.stage = 2
        st.rerun() 
    elif st.sidebar.button("Model Hyper Parameter Tunning"):
        st.session_state.stage = 3
        st.rerun() 
    elif st.sidebar.button("Model Prediction"):
        st.session_state.stage = 4
        st.rerun() 
    st.subheader("Regression Model Mean Absolute Error")

    
    accuracies2 = load_train_models2()
    accuracy_df = pd.DataFrame(accuracies2)

    # Find the min and max MAE
    min_mae = accuracy_df["MAE"].min()
    max_mae = accuracy_df["MAE"].max()

    # Apply highlighting to the DataFrame
    def highlight_mae(row):
        if row["MAE"] == min_mae:
            return ['background-color: green; color: white;'] * len(row)
        
        else:
            return [''] * len(row)

    # Style the DataFrame
    styled_df = accuracy_df.style.apply(highlight_mae, axis=1)

    # Display the table with a title
    st.write("Table 1. Machine Learning MAE Results")
    st.dataframe(styled_df)
    accuracies2 = load_train_models2()

    if "accuracies2" not in st.session_state:
        st.error("No accuracy data found. Please go back to Stage 1.")
        return  # Exit if accuracies are not available+
    
    
    
# Get the stored accuracies from session state
    accuracies2 = st.session_state.accuracies2

    # Extract numeric accuracies and find the lowest and highest one
    try:
        accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
        
        # Find the lowest accuracy
        lowest_accuracy = min(accuracies_numeric)
        lowest_model_index = accuracies_numeric.index(lowest_accuracy)
        lowest_model_name = accuracies2[lowest_model_index]["Model"]
        
        # Find the highest accuracy
        highest_accuracy = max(accuracies_numeric)
        highest_model_index = accuracies_numeric.index(highest_accuracy)
        highest_model_name = accuracies2[highest_model_index]["Model"]
        
        # Create the justified text with color highlights
        message = f"""
        <div style="text-align: justify;">
            The table highlights 10 Machine Learning algorithms with Mean Absolute Error (MAE) results. 
            The <span style="color: green; font-weight: bold;">{lowest_model_name}</span> yields the lowest MAE 
            (<span style="color: green; font-weight: bold;">{lowest_accuracy}</span>), showcasing high accuracy. 
            Meanwhile, the <span style="color: red; font-weight: bold;">{highest_model_name}</span> has the highest MAE 
            (<span style="color: red; font-weight: bold;">{highest_accuracy}</span>), suggesting it has the lowest accuracy 
            among the 10 models trained.
        </div>
        """
        
        # Display the styled message
        st.markdown(message, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
# Stage 2: Show the accuracy comparison as a bar chart
import plotly.graph_objects as go
def stage2():
    
    # Load model accuracies
    accuracies2 = load_train_models2()
    st.sidebar.title("NAVIGATION")

    if st.sidebar.button("Model Table Summary"):
        st.session_state.stage = 1
        st.rerun() 
    elif st.sidebar.button("Model Comparison Graph"):
        
        st.rerun() 
    elif st.sidebar.button("Model Hyper Parameter Tunning"):
        st.session_state.stage = 3
        st.rerun() 
    elif st.sidebar.button("Model Prediction"):
        st.session_state.stage = 4
        st.rerun() 
    st.markdown("<h1 style='text-align: center;'>Sea Level Regression Model Tester and Trainer</h1>", unsafe_allow_html=True)
   
    if "accuracies2" not in st.session_state:
        st.error("No accuracy data found. Please go back to Stage 1.")
        return  
    
    accuracies2 = st.session_state.accuracies2

    #Bar Graph
    models = [accuracy["Model"] for accuracy in accuracies2]
    accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
    # Create a Plotly bar chart
    max_mae = max(accuracies_numeric)
    min_mae = min(accuracies_numeric)

    # Create a color list for the bars
    colors = ['red' if mae == max_mae else 'green' if mae == min_mae else 'skyblue' for mae in accuracies_numeric]

    fig = go.Figure()

    # Add the bar chart
    fig.add_trace(go.Bar(
        x=models,
        y=accuracies_numeric,
        marker=dict(color=colors),
        hoverinfo='x+y',  # This ensures only model and MAE are shown in hover
    ))

    # Set title and labels with larger font sizes
    fig.update_layout(
        title="Model Accuracy Comparison",
        title_font=dict(size=30, color='white'),
        title_x=0.25,  # Centers the title
        xaxis_title="Model",
        xaxis_title_font=dict(size=30, color='white'),
        yaxis_title="MAE",
        yaxis_title_font=dict(size=30, color='white'),
        xaxis=dict(tickangle=25, tickfont=dict(size=15, color='white')),
         yaxis=dict(
        tickfont=dict(size=15, color='white'),
        tickvals=[0, 5, 10, 15, 20, 25],  # Custom y-axis tick values
        range=[0, 25],  
    ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        width=1400,  # Set the width of the figure
        height=600,  # Set the height of the figure
    )

    # Make the bars zoom in when hovered by adding hover effects
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Button to go back to Stage 1
    
    st.markdown("<h5 style='text-align: center;'>Figure 1. Bar Chart of the Model Comparison of the MAE Results</h5>", unsafe_allow_html=True)
    try:
        accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
        
        # Find the lowest accuracy
        lowest_accuracy = min(accuracies_numeric)
        lowest_model_index = accuracies_numeric.index(lowest_accuracy)
        lowest_model_name = accuracies2[lowest_model_index]["Model"]
        
        # Find the highest accuracy
        highest_accuracy = max(accuracies_numeric)
        highest_model_index = accuracies_numeric.index(highest_accuracy)
        highest_model_name = accuracies2[highest_model_index]["Model"]
        
        # Create the justified text with color highlights
        message = f"""
        <div style="text-align: justify;">
            The table highlights 10 Machine Learning algorithms with Mean Absolute Error (MAE) results. 
            The <span style="color: green; font-weight: bold;">{lowest_model_name}</span> yields the lowest MAE 
            (<span style="color: green; font-weight: bold;">{lowest_accuracy} </span>), showcasing high accuracy. 
            Meanwhile, the <span style="color: red; font-weight: bold;">{highest_model_name}</span> has the highest MAE 
            (<span style="color: red; font-weight: bold;">{highest_accuracy} </span>), suggesting it has the lowest accuracy 
            among the 10 models trained.
        </div>
        """
        
        # Display the styled message
        st.markdown(message, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
     # Button to proceed to Stage 2
   
def stage3():
    st.markdown("<h1 style='text-align: center;'>Sea Level Regression Model Tester and Trainer</h1>", unsafe_allow_html=True)
    st.title("Model Hyper Parameter Tuning")
   
    accuracies2 = load_train_models2()
    st.sidebar.title("NAVIGATION")
    if st.sidebar.button("Model Table Summary"):
        st.session_state.stage = 1
        st.rerun() 
    elif st.sidebar.button("Model Comparison Graph"):
        st.session_state.stage = 2
        st.rerun() 
    elif st.sidebar.button("Model Hyper Parameter Tunning"):
        
        st.rerun() 
    elif st.sidebar.button("Model Prediction"):
        st.session_state.stage = 4
        st.rerun() 
    if "accuracies2" not in st.session_state:
        st.error("No accuracy data found. Please go back to Stage 1.")
        return  # Exit if accuracies are not available+
    
    # Get the stored accuracies from session state
    accuracies2 = st.session_state.accuracies2
    
    # Extract numeric accuracies and find the highest one
    accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]

# Find the lowest accuracy
    lowest_accuracy = min(accuracies_numeric)
    lowest_model_index = accuracies_numeric.index(lowest_accuracy)


    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]
#Linear Regression
#Adaboost
#Lasso Regression
#Ridge Regression
#MLP Regressor
#Random Forest Regressor
#SVM Regressor
#Elastic Net
#Decision Tree
#KNN Regressor
    if lowest_model_index == 0:  # Decision Tree
        st.session_state.selected_model = "lr"
        lr_tuning(X, y)
    elif lowest_model_index == 1:  # Decision Tree
        st.session_state.selected_model = "gbm"
        gbm_tuning(X, y)
    elif lowest_model_index == 2:  # Decision Tree
        st.session_state.selected_model = "Lasso"
        lasso_tuning(X, y)
    elif lowest_model_index == 3:  # Decision Tree
        st.session_state.selected_model = "Ridge"
        ridge_tuning(X, y)
    elif lowest_model_index == 4:  # Decision Tree
        st.session_state.selected_model = "mlp"
        mlp_tuning(X, y)
    elif lowest_model_index == 5:  # Decision Tree
        st.session_state.selected_model = "rft"
        rft_tuning(X, y)
    elif lowest_model_index == 6:  # Decision Tree
        st.session_state.selected_model = "svm"
        svm_tuning(X, y)
    elif lowest_model_index == 7:  # Decision Tree
        st.session_state.selected_model = "elastic"
        elastic_tuning(X, y)
    elif lowest_model_index == 8:  # Decision Tree
        st.session_state.selected_model = "dt"
        dt_tuning(X, y)
    else :  # Gaussian NB
        st.session_state.selected_model = "knn"
        knn_tuning(X, y)

def lr_tuning(X, y):
    st.subheader("Linear Regression")
    
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            
            st.sidebar.subheader("Hyperparameter Tuning")
            
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=123)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            

            #LR
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Linear Regression, with a mean Accuracy: {mae:.2f}")
                
            #gbm
            
                


    
    perform_train_split(X, y, model, "lr")
def gbm_tuning(X, y):
    st.subheader("gbm")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=42)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            
            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=random_seed
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Gradient Boosting Machine, with a mean Accuracy: {mae:.2f}")
                
            
            
            

    
    perform_train_split(X, y, model, "gbm")


#Lasso Regression
def lasso_tuning(X, y):
    st.subheader("lasso")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=42)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            #DT
            alpha_slider = st.slider("Alpha (Lasso Regularization)", min_value=0.01, max_value=100.0, value=0.1, step=0.01)

            # Initialize and train Lasso Regressor
            model = Lasso(alpha=alpha_slider)
            model.fit(X_train, y_train)

            # Make predictions and calculate MAE
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            # Display results
            st.subheader("Lasso Model Performance")
            st.write(f"Using Alpha = {alpha_slider:.2f}, the Mean Absolute Error (MAE) is: {mae:.2f}")

            # Conclusion
            st.subheader("Conclusion")
            st.write(f"The model with the highest accuracy is: Lasso, with a mean MAE of {mae:.2f}")
                            
            
            

    
    perform_train_split(X, y, model, "laso")

#Ridge Regression

def ridge_tuning(X, y):
    st.subheader("ridge")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=42)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            # Ridge Regression model
            
# List of alpha values to try for Ridge regression
            alpha_values = [0.01, 0.1, 1, 10, 100]

            # Initialize variables to store the best model
            best_alpha = None
            best_mae = float('inf')
            best_model = None

            # Loop through each alpha value
            for alpha in alpha_values:
                # Initialize the Ridge regressor with the current alpha value
                model = Ridge(alpha=alpha)
                
                # Fit the model to the training data
                model.fit(X_train, y_train)
                
                # Make predictions on the test set
                y_pred = model.predict(X_test)
                
                # Calculate Mean Absolute Error (MAE)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Check if this model has the lowest MAE so far
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
                    best_model = model

                        
                st.subheader("Conclusion")
            
                st.write(f"The model with the highest accuracy is :Ridge , with a mean Accuracy: {mae:.2f}")
                
            #mlp
            

            

    
    perform_train_split(X, y, model, "ridge")
#MLP Regressor
def mlp_tuning(X, y):
    st.subheader("mlp")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
    
# Assuming 'data' is already loaded
        X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
        y = data["SmoothedGSML_GIA_sigremoved"]

        # User inputs for train-test split and random seed
        train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
        random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
        test_size = 1 - train_size

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)

        # Hyperparameter sliders for MLPRegressor
        hidden_layer_sizes = st.sidebar.slider("Hidden Layer Sizes", min_value=10, max_value=500, value=100, step=10)
        max_iter = st.sidebar.slider("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
        alpha = st.sidebar.slider("Alpha (L2 Regularization)", min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001)
        learning_rate_init = st.sidebar.slider("Learning Rate (Initial)", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001)

        # Initialize and train MLPRegressor with selected hyperparameters
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_sizes,), 
            max_iter=max_iter, 
            alpha=alpha, 
            learning_rate_init=learning_rate_init,
            random_state=random_seed
        )

        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred =model.predict(X_test)

        # Evaluate the model performance using MAE and MSE
        mae = mean_absolute_error(y_test, y_pred)

            
        st.subheader("Conclusion")
            
        st.write(f"The model with the highest accuracy is :Multi Layer Perceptron, with a mean Accuracy: {mae:.2f}")
                
    
    perform_train_split(X, y, model, "mlp")

#Random Forest Regressor
def rft_tuning(X, y):
    
    st.subheader("Random Foret Tree Hyper Parameter Tuning")
    if st.button("Original Review Table Summary"):
        accuracies2 = st.session_state.accuracies2

    #Bar Graph
        models = [accuracy["Model"] for accuracy in accuracies2]
        accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
        # Create a Plotly bar chart
        max_mae = max(accuracies_numeric)
        min_mae = min(accuracies_numeric)

        # Create a color list for the bars
        colors = ['red' if mae == max_mae else 'green' if mae == min_mae else 'skyblue' for mae in accuracies_numeric]

        fig = go.Figure()

        # Add the bar chart
        fig.add_trace(go.Bar(
            x=models,
            y=accuracies_numeric,
            marker=dict(color=colors),
            hoverinfo='x+y',  # This ensures only model and MAE are shown in hover
        ))

        # Set title and labels with larger font sizes
        fig.update_layout(
            title="Model Accuracy Comparison",
            title_font=dict(size=30, color='white'),
            title_x=0.25,  # Centers the title
            xaxis_title="Model",
            xaxis_title_font=dict(size=30, color='white'),
            yaxis_title="MAE",
            yaxis_title_font=dict(size=30, color='white'),
            xaxis=dict(tickangle=25, tickfont=dict(size=15, color='white')),
            yaxis=dict(
            tickfont=dict(size=15, color='white'),
            tickvals=[0, 5, 10, 15, 20, 25],  # Custom y-axis tick values
            range=[0, 25],  
        ),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            width=1400,  # Set the width of the figure
            height=600,  # Set the height of the figure
        )
        st.plotly_chart(fig, use_container_width=True)
        # Make the bars zoom in when hovered by adding hover effects
        if st.button("close"):
            st.plotly_chart(fig, use_container_width=False)
            # Display the Plotly figure in Streamlit
        
    
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            
            
            col1, col2 = st.columns([1, 1])

            # Number of Estimators input in the first column
            with col1:
                st.markdown("<p style='font-size:12px;margin-top:30px'>"
                " </p>", 
                unsafe_allow_html=True)
                n_estimators = st.slider("Number of Estimators", min_value=10, max_value=500, value=100, step=10)

            # Description beside the input in the second column
            with col2:
                st.markdown("<p style='font-size:12px;margin-top:60px'>"
                            "The number of decision trees in the Random Forest. "
                            "More trees enhance performance but increase computation time.</p>", 
                            unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])

            # Training Set Size input in the first column
            with col1:
                st.markdown("<p style='font-size:12px;margin-top:30px'>"" </p>", 
                unsafe_allow_html=True)
                train_size = st.slider("Training Set Size", min_value=0.1, max_value=1.00, value=0.8,step=0.1)

            # Description beside the input in the second column
            with col2:
                st.markdown("<p style='font-size:12px;margin-top:60px'>"
                            "The proportion of the dataset used for training the model. "
                            "It can be a float (e.g., 0.8 for 80%) or an integer (number of samples). "
                            "The rest is used for testing.</p>", 
                            unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])

            # Random Seed input in the first column
            with col1:
                st.markdown("<p style='font-size:12px;margin-top:30px'>"
                " </p>", 
                unsafe_allow_html=True)
                random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=123)

            # Description beside the input in the second column
            with col2:
                st.markdown("<p style='font-size:12px;margin-top:60px;'>"
                            "A value used to initialize the random number generator. "
                            "It ensures reproducibility by making the results of random operations consistent across runs.</p>", 
                            unsafe_allow_html=True)

            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            # Initialize and train AdaBoost Regressor
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)  # You can tune the number of trees (n_estimators)
         
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.markdown("<p style='font-size:12px;margin-top:30px'>"
                " </p>", 
                unsafe_allow_html=True)
            st.subheader("Conclusion")
            
            st.markdown(f"The model with the highest accuracy is: <span style='color:green;'>Random Forest</span> , with a mean Accuracy : <span style='color:green;'> {mae:.2f}</span>", unsafe_allow_html=True)
    
    perform_train_split(X, y, model, "rft")
    

#SVM Regressor

def svm_tuning(X, y):
    st.subheader("svm")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
          "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            # User inputs for train-test split and random seed
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)

            # --- Hyperparameter tuning for SVM Regressor ---
            svm_kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "poly", "rbf", "sigmoid"])
            svm_C = st.sidebar.slider("SVM C", min_value=0.01, max_value=1000.0, value=10.0, step=0.01)
            svm_epsilon = st.sidebar.slider("SVM Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

            # Initialize and train the SVM Regressor
            model = SVR(kernel=svm_kernel, C=svm_C, epsilon=svm_epsilon)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            svm_mae = mean_absolute_error(y_test, y_pred)
            svm_mse = mean_squared_error(y_test, y_pred)
            
            
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Support Vector Machines, with a mean Accuracy: {svm_mae:.2f}")
                
    
    perform_train_split(X, y, model, "svm")

#Elastic Net

def elastic_tuning(X, y):
    st.subheader("elastic")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            elastic_alpha = st.sidebar.slider("ElasticNet Alpha", min_value=0.001, max_value=10.0, value=0.1, step=0.001)
            elastic_l1_ratio = st.sidebar.slider("ElasticNet L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            # Initialize and train ElasticNet
            model = ElasticNet(alpha=elastic_alpha, l1_ratio=elastic_l1_ratio)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elastic_mae = mean_absolute_error(y_test, y_pred)
            elastic_mse = mean_squared_error(y_test, y_pred)
            
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Elastic Net, with a mean Accuracy: {elastic_mae:.2f}%")
                

    
    perform_train_split(X, y, model, "elastic")
#Decision Tree

def dt_tuning(X, y):
    st.subheader("dt")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            dt_max_depth = st.sidebar.slider("Decision Tree Max Depth", min_value=1, max_value=50, value=10, step=1)
            dt_min_samples_split = st.sidebar.slider("Decision Tree Min Samples Split", min_value=2, max_value=10, value=2, step=1)

            model = DecisionTreeRegressor(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, random_state=random_seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            dt_mae = mean_absolute_error(y_test, y_pred)
            dt_mse = mean_squared_error(y_test, y_pred)
                        
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Decision Tree, with a mean Accuracy: {dt_mae:.2f}%")
                

    
    perform_train_split(X, y, model, "dt")
#KNN Regressor
def knn_tuning(X, y):
    st.subheader("knn")
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
        

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
    required_columns = ["Year","TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                            "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA", "SmoothedGSML_GIA_sigremoved"]

    if all(col in data.columns for col in required_columns):
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            # --- Hyperparameter tuning for KNN Regressor ---
            knn_neighbors = st.sidebar.slider("KNN Neighbors", min_value=1, max_value=50, value=5, step=1)
            knn_weights = st.sidebar.selectbox("KNN Weights", ["uniform", "distance"])

            # Initialize and train KNN Regressor
            model = KNeighborsRegressor(n_neighbors=knn_neighbors, weights=knn_weights)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            knn_mae = mean_absolute_error(y_test, y_pred)
            knn_mse = mean_squared_error(y_test, y_pred)
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :K Nearest Neighbors, with a mean Accuracy: {knn_mae:.2f}%")
                    
            

    
    perform_train_split(X, y, model, "knn")
def perform_train_split(X, y, model, model_name):
    st.markdown("<p style='font-size:12px;margin-top:30px'>"
                " </p>", 
                unsafe_allow_html=True)
    model_filename = "regressionmodel.joblib"
    joblib.dump(model, model_filename)

    with open(model_filename, "rb") as f:
        model_data = f.read()

    st.download_button(
        label="Download Trained Model",
        data=model_data,
        file_name=model_filename,
        mime="application/octet-stream"
    )
    

    

def stage4():
    
    # Load model accuracies 
    
    st.title("Sea Level Prediction Application")

    

    # Sidebar navigation
    st.sidebar.title("NAVIGATION")

    # Buttons with custom style
    if st.sidebar.button("Model Table Summary"):
        st.session_state.stage = 1
        st.rerun() 

    elif st.sidebar.button("Model Comparison Graph"):
        st.session_state.stage = 2
        st.rerun() 

    elif st.sidebar.button("Model Hyper Parameter Tuning"):
        st.session_state.stage = 3
        st.rerun() 

    elif st.sidebar.button("Model Prediction"):
        st.session_state.stage = 4
        st.rerun() 
        
    
   
    # Step 1: Upload the pre-trained model
    csv_file_path2 = "C:/Users/Admin/Documents/ITE105/LabFinal/regressionmodel.joblib"
    try:
        model = joblib.load(csv_file_path2)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Input form for user data
        st.success("Model loaded successfully!")

            # Step 2: Input data for prediction
        st.subheader("Input Data for Prediction")

        # Input fields for each feature
    
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        year = st.date_input("Select a Year", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))

# Extract the year from the selected date
        year = year.year

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                   "Refer to the Year of the Observation", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        TotalWeightedObservations = st.number_input("Total Weighted Observations", value=0.0, format="%.2f")

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                   "Refer to the total number of observations made in a year"
                   " Mostly more than 3,000+ observations yearly", 
                    unsafe_allow_html=True)
        
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        GMSL_noGIA = st.number_input("GMSL noGIA", value=0.0, format="%.2f")
    

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to sea level changes in (mm) with respect to 20-year TOPEX/Jason collinear mean reference:</p>", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        StdDevGMSL_noGIA = st.number_input("Standard Deviation of GMSL noGIA", value=0.0, format="%.2f")
    

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to the standard deviation of  the mean of the sea level changes in (mm)  </p>", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        SmoothedGSML_noGIA = st.number_input("Smoothed  GMSL noGIA", value=0.0, format="%.2f")
    

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to the 60-day Gaussian type filter of sea level changes in (mm) </p>", 
                    unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        GMSL_GIA = st.number_input("GMSL GIA", value=0.0, format="%.2f")
    

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    "Refers to the Global Mean Sea Level changes affected by the Glacial Isostatic Adjustments in (mm)</p>", # with respect to 20-year TOPEX/Jason collinear mean reference 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        StdDevGMSL_GIA = st.number_input("Standard Deviation of GMSL (GIA applied)", value=0.0, format="%.2f")
    

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:25px;'>"
                    "Refers to the standard deviation of  the mean of the sea level changes affected by Glacial Isostatic Adjustments in (mm)</p>", # with respect to 20-year TOPEX/Jason collinear mean reference 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            # Random Seed input in the first column
    with col1:
        SmoothedGSML_GIA = st.number_input("Smoothed GMSL GIA", value=0.0, format="%.2f")
    

    # Description beside the input in the second column
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    "Refers to the 60-day Gaussian type filter of sea level changes affected by Global Isostatic Adjustments in (mm)</p>", # with respect to 20-year TOPEX/Jason collinear mean reference 
                    unsafe_allow_html=True)
    
    
    

    # Organize inputs into a single feature array
    input_features = np.array([[TotalWeightedObservations, GMSL_noGIA, StdDevGMSL_noGIA,
                                SmoothedGSML_noGIA, GMSL_GIA, StdDevGMSL_GIA, SmoothedGSML_GIA]])

    # Step 3: Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_features)[0]  # Get the prediction
        st.subheader("Prediction Result")
        st.write(f"The Predicted Sea Rise Level by  {int(year):d} is  {prediction:.2f} in millimeters")

    





if "stage" not in st.session_state:
    st.session_state.stage = 1  # Default to Stage 1

if "accuracies" not in st.session_state:
    st.session_state.accuracies = load_train_models()  # Load and cache the accuracies if not already done

if "accuracies2" not in st.session_state:
    st.session_state.accuracies2 = load_train_models2()  # Load and cache the accuracies if not already done

if st.session_state.stage == 1:
    stage1()  # Show Stage 1
elif st.session_state.stage == 2:
    stage2()  # Show Stage 2
elif st.session_state.stage == 3:
    stage3()  # Show Stage 3 with Hyperparameter Tuning
elif st.session_state.stage == 4:
    stage4()  # Show Stage 3 with Hyperparameter Tuning
def reset_cache():
    st.cache_data.clear()  # Clears data cache
    st.cache_resource.clear()  # Clears resource cache
    st.session_state.clear()  # Clears session state
    st.experimental_rerun()  # Trigger a rerun to reset the state
    stage1()


if st.button("Reset"):
    reset_cache()  # Reset the cache when the button is clicked
    stage1()
    