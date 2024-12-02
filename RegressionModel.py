
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
from sklearn.tree import DecisionTreeClassifier  
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


@st.cache_resource
def load_train_models():
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
       
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
         
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]
            

            accuracies=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, train_size=0.8)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Linear Regression", "MAE": f"{mae:.2f}"})
            ada_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
            ada_regressor.fit(X_train, y_train)
            y_pred = ada_regressor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Adaboost", "MAE": f"{mae:.2f}"})
            lasso_regressor = Lasso(alpha=0.1)             
            lasso_regressor.fit(X_train, y_train)
            y_pred = lasso_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Lasso Regression", "MAE": f"{mae:.2f}"})

            ridge_regressor = Ridge(alpha=0.1)             
            ridge_regressor.fit(X_train, y_train)
            y_pred = ridge_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Ridge Regression", "MAE": f"{mae:.2f}"})

            mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)             
            mlp_regressor.fit(X_train, y_train)
            y_pred = mlp_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "MLP Regressor", "MAE": f"{mae:.2f}"})
            rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)            
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Random Forest Regressor", "MAE": f"{mae:.2f}"})

            svm_regressor = SVR(kernel='poly', C=10, epsilon=0.1)
            svm_regressor.fit(X_train, y_train)
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "SVM Regressor", "MAE": f"{mae:.2f}"})

                       
            elastic_net_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5)             
            elastic_net_regressor.fit(X_train, y_train)
            y_pred = elastic_net_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Elastic Net", "MAE": f"{mae:.2f}"})

            decision_tree_regressor = DecisionTreeRegressor(random_state=42)             
            decision_tree_regressor.fit(X_train, y_train)
            y_pred = decision_tree_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Decision Tree", "MAE": f"{mae:.2f}"})

            knn_regressor = KNeighborsRegressor(n_neighbors=5)             
            knn_regressor.fit(X_train, y_train)
            y_pred = knn_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "KNN Regressor", "MAE": f"{mae:.2f}"})
    
            return accuracies

@st.cache_resource
def load_train_models2():
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
         
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            accuracies2=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, train_size=0.8)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
          
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Linear Regression", "MAE": f"{mae:.2f}"})
            ada_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
            ada_regressor.fit(X_train, y_train)
            y_pred = ada_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Adaboost", "MAE": f"{mae:.2f}"})
            lasso_regressor = Lasso(alpha=0.1)             
            lasso_regressor.fit(X_train, y_train)
            y_pred = lasso_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Lasso Regression", "MAE": f"{mae:.2f}"})

            ridge_regressor = Ridge(alpha=0.1)             
            ridge_regressor.fit(X_train, y_train)
            y_pred = ridge_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Ridge Regression", "MAE": f"{mae:.2f}"})
            mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)             
            mlp_regressor.fit(X_train, y_train)
            y_pred = mlp_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "MLP Regressor", "MAE": f"{mae:.2f}"})
            rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)             
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Random Forest Regressor", "MAE": f"{mae:.2f}"})

            svm_regressor = SVR(kernel='poly', C=10, epsilon=0.1)            
            svm_regressor.fit(X_train, y_train)
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "SVM Regressor", "MAE": f"{mae:.2f}"})
            elastic_net_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5)             
            elastic_net_regressor.fit(X_train, y_train)
            y_pred = elastic_net_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Elastic Net", "MAE": f"{mae:.2f}"})
            decision_tree_regressor = DecisionTreeRegressor(random_state=42)             
            decision_tree_regressor.fit(X_train, y_train)
            y_pred = decision_tree_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Decision Tree", "MAE": f"{mae:.2f}"})
            knn_regressor = KNeighborsRegressor(n_neighbors=5)             
            knn_regressor.fit(X_train, y_train)
            y_pred = knn_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "KNN Regressor", "MAE": f"{mae:.2f}"})
            
            return accuracies2
    


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

    min_mae = accuracy_df["MAE"].min()
    max_mae = accuracy_df["MAE"].max()

    def highlight_mae(row):
        if row["MAE"] == min_mae:
            return ['background-color: green; color: white;'] * len(row)
        
        else:
            return [''] * len(row)

    styled_df = accuracy_df.style.apply(highlight_mae, axis=1)

    st.write("Table 1. Machine Learning MAE Results")
    st.dataframe(styled_df)
    accuracies2 = load_train_models2()

    if "accuracies2" not in st.session_state:
        st.error("No accuracy data found. Please go back to Stage 1.")
        return     
    
    
    accuracies2 = st.session_state.accuracies2

    try:
        accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
        
        
        lowest_accuracy = min(accuracies_numeric)
        lowest_model_index = accuracies_numeric.index(lowest_accuracy)
        lowest_model_name = accuracies2[lowest_model_index]["Model"]
        
        highest_accuracy = max(accuracies_numeric)
        highest_model_index = accuracies_numeric.index(highest_accuracy)
        highest_model_name = accuracies2[highest_model_index]["Model"]
        
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
        
        st.markdown(message, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
import plotly.graph_objects as go
def stage2():
    
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

    models = [accuracy["Model"] for accuracy in accuracies2]
    accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
    max_mae = max(accuracies_numeric)
    min_mae = min(accuracies_numeric)

    colors = ['red' if mae == max_mae else 'green' if mae == min_mae else 'skyblue' for mae in accuracies_numeric]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models,
        y=accuracies_numeric,
        marker=dict(color=colors),
        hoverinfo='x+y',     ))

    fig.update_layout(
        title="Model Accuracy Comparison",
        title_font=dict(size=30, color='white'),
        title_x=0.25,         xaxis_title="Model",
        xaxis_title_font=dict(size=30, color='white'),
        yaxis_title="MAE",
        yaxis_title_font=dict(size=30, color='white'),
        xaxis=dict(tickangle=25, tickfont=dict(size=15, color='white')),
         yaxis=dict(
        tickfont=dict(size=15, color='white'),
        tickvals=[0, 5, 10, 15, 20, 25],         range=[0, 25],  
    ),
        plot_bgcolor='rgba(0,0,0,0)',         paper_bgcolor='rgba(0,0,0,0)',         width=1400,         height=600,     )


    
   
    st.plotly_chart(fig, use_container_width=True)

    
    
    st.markdown("<h5 style='text-align: center;'>Figure 1. Bar Chart of the Model Comparison of the MAE Results</h5>", unsafe_allow_html=True)
    try:
        accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
        
        
        lowest_accuracy = min(accuracies_numeric)
        lowest_model_index = accuracies_numeric.index(lowest_accuracy)
        lowest_model_name = accuracies2[lowest_model_index]["Model"]
        
      
        highest_accuracy = max(accuracies_numeric)
        highest_model_index = accuracies_numeric.index(highest_accuracy)
        highest_model_name = accuracies2[highest_model_index]["Model"]
        
     
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
        
      
        st.markdown(message, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
     
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
        return  

    accuracies2 = st.session_state.accuracies2
    
 
    accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]


    lowest_accuracy = min(accuracies_numeric)
    lowest_model_index = accuracies_numeric.index(lowest_accuracy)


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
         
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

    if lowest_model_index == 0:         
        st.session_state.selected_model = "lr"
        lr_tuning(X, y)
    elif lowest_model_index == 1:         
        st.session_state.selected_model = "gbm"
        gbm_tuning(X, y)
    elif lowest_model_index == 2:         
        st.session_state.selected_model = "Lasso"
        lasso_tuning(X, y)
    elif lowest_model_index == 3:         
        st.session_state.selected_model = "Ridge"
        ridge_tuning(X, y)
    elif lowest_model_index == 4:         
        st.session_state.selected_model = "mlp"
        mlp_tuning(X, y)
    elif lowest_model_index == 5:         
        st.session_state.selected_model = "rft"
        rft_tuning(X, y)
    elif lowest_model_index == 6:         
        st.session_state.selected_model = "svm"
        svm_tuning(X, y)
    elif lowest_model_index == 7:         
        st.session_state.selected_model = "elastic"
        elastic_tuning(X, y)
    elif lowest_model_index == 8:         
        st.session_state.selected_model = "dt"
        dt_tuning(X, y)
    else :         
        st.session_state.selected_model = "knn"
        knn_tuning(X, y)

def lr_tuning(X, y):
    st.subheader("Linear Regression")
    
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
         
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            
            st.sidebar.subheader("Hyperparameter Tuning")
            
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=123)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            

            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Linear Regression, with a mean Accuracy: {mae:.2f}")
                
     
                


    
    perform_train_split(X, y, model, "lr")
def gbm_tuning(X, y):
    st.subheader("gbm")
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


def lasso_tuning(X, y):
    st.subheader("lasso")
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
          
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=42)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            
            alpha_slider = st.slider("Alpha (Lasso Regularization)", min_value=0.01, max_value=100.0, value=0.1, step=0.01)

           
            model = Lasso(alpha=alpha_slider)
            model.fit(X_train, y_train)
         
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

           
            st.subheader("Lasso Model Performance")
            st.write(f"Using Alpha = {alpha_slider:.2f}, the Mean Absolute Error (MAE) is: {mae:.2f}")

        
            st.subheader("Conclusion")
            st.write(f"The model with the highest accuracy is: Lasso, with a mean MAE of {mae:.2f}")
                            
            
            

    
    perform_train_split(X, y, model, "laso")



def ridge_tuning(X, y):
    st.subheader("ridge")
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
      
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=42)
            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
       
            alpha_values = [0.01, 0.1, 1, 10, 100]

            best_alpha = None
            best_mae = float('inf')
            best_model = None

      
            for alpha in alpha_values:
          
                model = Ridge(alpha=alpha)
            
                model.fit(X_train, y_train)
                
              
                y_pred = model.predict(X_test)
                
             
                mae = mean_absolute_error(y_test, y_pred)
                
               
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
                    best_model = model

                        
                st.subheader("Conclusion")
            
                st.write(f"The model with the highest accuracy is :Ridge , with a mean Accuracy: {mae:.2f}")
             
            

    
    perform_train_split(X, y, model, "ridge")
def mlp_tuning(X, y):
    st.subheader("mlp")
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
    

        X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
        y = data["SmoothedGSML_GIA_sigremoved"]

    
        train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
        random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
        test_size = 1 - train_size

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)

    
        hidden_layer_sizes = st.sidebar.slider("Hidden Layer Sizes", min_value=10, max_value=500, value=100, step=10)
        max_iter = st.sidebar.slider("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
        alpha = st.sidebar.slider("Alpha (L2 Regularization)", min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001)
        learning_rate_init = st.sidebar.slider("Learning Rate (Initial)", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001)

    
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_sizes,), 
            max_iter=max_iter, 
            alpha=alpha, 
            learning_rate_init=learning_rate_init,
            random_state=random_seed
        )

        model.fit(X_train, y_train)

     
        y_pred =model.predict(X_test)

      
        mae = mean_absolute_error(y_test, y_pred)

            
        st.subheader("Conclusion")
            
        st.write(f"The model with the highest accuracy is :Multi Layer Perceptron, with a mean Accuracy: {mae:.2f}")
                
    
    perform_train_split(X, y, model, "mlp")

def rft_tuning(X, y):
    
    st.subheader("Random Foret Tree Hyper Parameter Tuning")
    if st.button("Original Review Table Summary"):
        accuracies2 = st.session_state.accuracies2

        models = [accuracy["Model"] for accuracy in accuracies2]
        accuracies_numeric = [float(accuracy["MAE"].rstrip('')) for accuracy in accuracies2]
        
        max_mae = max(accuracies_numeric)
        min_mae = min(accuracies_numeric)

      
        colors = ['red' if mae == max_mae else 'green' if mae == min_mae else 'skyblue' for mae in accuracies_numeric]

        fig = go.Figure()

    
        fig.add_trace(go.Bar(
            x=models,
            y=accuracies_numeric,
            marker=dict(color=colors),
            hoverinfo='x+y',  
        ))

       
        fig.update_layout(
            title="Model Accuracy Comparison",
            title_font=dict(size=30, color='white'),
            title_x=0.25, 
            xaxis_title="Model",
            xaxis_title_font=dict(size=30, color='white'),
            yaxis_title="MAE",
            yaxis_title_font=dict(size=30, color='white'),
            xaxis=dict(tickangle=25, tickfont=dict(size=15, color='white')),
            yaxis=dict(
            tickfont=dict(size=15, color='white'),
            tickvals=[0, 5, 10, 15, 20, 25],
            range=[0, 25],  
        ),
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            width=1400, 
            height=600,  
        )
        st.plotly_chart(fig, use_container_width=True)
   
        if st.button("close"):
            st.plotly_chart(fig, use_container_width=False)
         
        
    
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
         
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            
            
            col1, col2 = st.columns([1, 1])

        
            with col1:
                st.markdown("<p style='font-size:12px;margin-top:30px'>"
                " </p>", 
                unsafe_allow_html=True)
                n_estimators = st.slider("Number of Estimators", min_value=10, max_value=500, value=100, step=10)

      
            with col2:
                st.markdown("<p style='font-size:12px;margin-top:60px'>"
                            "The number of decision trees in the Random Forest. "
                            "More trees enhance performance but increase computation time.</p>", 
                            unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])

         
            with col1:
                st.markdown("<p style='font-size:12px;margin-top:30px'>"" </p>", 
                unsafe_allow_html=True)
                train_size = st.slider("Training Set Size", min_value=0.1, max_value=1.00, value=0.8,step=0.1)

            
            with col2:
                st.markdown("<p style='font-size:12px;margin-top:60px'>"
                            "The proportion of the dataset used for training the model. "
                            "It can be a float (e.g., 0.8 for 80%) or an integer (number of samples). "
                            "The rest is used for testing.</p>", 
                            unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])

          
            with col1:
                st.markdown("<p style='font-size:12px;margin-top:30px'>"
                " </p>", 
                unsafe_allow_html=True)
                random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=123)

       
            with col2:
                st.markdown("<p style='font-size:12px;margin-top:60px;'>"
                            "A value used to initialize the random number generator. "
                            "It ensures reproducibility by making the results of random operations consistent across runs.</p>", 
                            unsafe_allow_html=True)

            test_size = 1 - train_size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
          
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)          
            model.fit(X_train, y_train)

         
            y_pred = model.predict(X_test)

           
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.markdown("<p style='font-size:12px;margin-top:30px'>"
                " </p>", 
                unsafe_allow_html=True)
            st.subheader("Conclusion")
            
            st.markdown(f"The model with the highest accuracy is: <span style='color:green;'>Random Forest</span> , with a mean Accuracy : <span style='color:green;'> {mae:.2f}</span>", unsafe_allow_html=True)
    
    perform_train_split(X, y, model, "rft")
    



def svm_tuning(X, y):
    st.subheader("svm")
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
        
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
          "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

         
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size

           
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)

        
            svm_kernel = st.sidebar.selectbox("SVM Kernel", ["linear", "poly", "rbf", "sigmoid"])
            svm_C = st.sidebar.slider("SVM C", min_value=0.01, max_value=1000.0, value=10.0, step=0.01)
            svm_epsilon = st.sidebar.slider("SVM Epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

      
            model = SVR(kernel=svm_kernel, C=svm_C, epsilon=svm_epsilon)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            svm_mae = mean_absolute_error(y_test, y_pred)
            svm_mse = mean_squared_error(y_test, y_pred)
            
            
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Support Vector Machines, with a mean Accuracy: {svm_mae:.2f}")
                
    
    perform_train_split(X, y, model, "svm")



def elastic_tuning(X, y):
    st.subheader("elastic")
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
         
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
            elastic_alpha = st.sidebar.slider("ElasticNet Alpha", min_value=0.001, max_value=10.0, value=0.1, step=0.001)
            elastic_l1_ratio = st.sidebar.slider("ElasticNet L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            model = ElasticNet(alpha=elastic_alpha, l1_ratio=elastic_l1_ratio)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elastic_mae = mean_absolute_error(y_test, y_pred)
            elastic_mse = mean_squared_error(y_test, y_pred)
            
            st.subheader("Conclusion")
            
            st.write(f"The model with the highest accuracy is :Elastic Net, with a mean Accuracy: {elastic_mae:.2f}%")
                

    
    perform_train_split(X, y, model, "elastic")


def dt_tuning(X, y):
    st.subheader("dt")
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
           
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size


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
def knn_tuning(X, y):
    st.subheader("knn")
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
          
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed", min_value=10, max_value=500, value=42)
            test_size = 1 - train_size

   
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, train_size=train_size)
         
            knn_neighbors = st.sidebar.slider("KNN Neighbors", min_value=1, max_value=50, value=5, step=1)
            knn_weights = st.sidebar.selectbox("KNN Weights", ["uniform", "distance"])

           
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
    
       
    st.title("Sea Level Prediction Application")

    

    st.sidebar.title("NAVIGATION")

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
        
    
   
   
    csv_file_path2 = "C:/Users/Admin/Documents/ITE105/LabFinal/regressionmodel.joblib"
    try:
        model = joblib.load(csv_file_path2)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

        st.success("Model loaded successfully!")

       
        st.subheader("Input Data for Prediction")

      
    
    col1, col2 = st.columns([4, 3])

    with col1:
        year = st.date_input("Select a Year", min_value=datetime(2000, 1, 1), max_value=datetime(2100, 12, 31))

        year = year.year

   
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                   "Refer to the Year of the Observation", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

          
    with col1:
        TotalWeightedObservations = st.number_input("Total Weighted Observations", value=0.0, format="%.2f")

    
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                   "Refer to the total number of observations made in a year"
                   " Mostly more than 3,000+ observations yearly", 
                    unsafe_allow_html=True)
        
    col1, col2 = st.columns([4, 3])


    with col1:
        GMSL_noGIA = st.number_input("GMSL noGIA", value=0.0, format="%.2f")
    

    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to sea level changes in (mm) with respect to 20-year TOPEX/Jason collinear mean reference:</p>", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            
    with col1:
        StdDevGMSL_noGIA = st.number_input("Standard Deviation of GMSL noGIA", value=0.0, format="%.2f")
    

    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to the standard deviation of  the mean of the sea level changes in (mm)  </p>", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

            
    with col1:
        SmoothedGSML_noGIA = st.number_input("Smoothed  GMSL noGIA", value=0.0, format="%.2f")
    

 
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    " Refers to the 60-day Gaussian type filter of sea level changes in (mm) </p>", 
                    unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 3])

       
    with col1:
        GMSL_GIA = st.number_input("GMSL GIA", value=0.0, format="%.2f")
    

  
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    "Refers to the Global Mean Sea Level changes affected by the Glacial Isostatic Adjustments in (mm)</p>",                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

 
    with col1:
        StdDevGMSL_GIA = st.number_input("Standard Deviation of GMSL (GIA applied)", value=0.0, format="%.2f")
    


    with col2:
        st.markdown("<p style='font-size:12px;margin-top:25px;'>"
                    "Refers to the standard deviation of  the mean of the sea level changes affected by Glacial Isostatic Adjustments in (mm)</p>",                    unsafe_allow_html=True)
    col1, col2 = st.columns([4, 3])

          
    with col1:
        SmoothedGSML_GIA = st.number_input("Smoothed GMSL GIA", value=0.0, format="%.2f")
    

 
    with col2:
        st.markdown("<p style='font-size:12px;margin-top:30px;'>"
                    "Refers to the 60-day Gaussian type filter of sea level changes affected by Global Isostatic Adjustments in (mm)</p>",                    unsafe_allow_html=True)
    
    
    

  
    input_features = np.array([[TotalWeightedObservations, GMSL_noGIA, StdDevGMSL_noGIA,
                                SmoothedGSML_noGIA, GMSL_GIA, StdDevGMSL_GIA, SmoothedGSML_GIA]])

   
    if st.button("Predict"):
        prediction = model.predict(input_features)[0]  
        st.subheader("Prediction Result")
        st.write(f"The Predicted Sea Rise Level by  {int(year):d} is  {prediction:.2f} in millimeters")

    





if "stage" not in st.session_state:
    st.session_state.stage = 1

if "accuracies" not in st.session_state:
    st.session_state.accuracies = load_train_models() 

if "accuracies2" not in st.session_state:
    st.session_state.accuracies2 = load_train_models2()  
if st.session_state.stage == 1:
    stage1()  
elif st.session_state.stage == 2:
    stage2()  
elif st.session_state.stage == 3:
    stage3()  
elif st.session_state.stage == 4:
    stage4()  
def reset_cache():
    st.cache_data.clear()  
    st.cache_resource.clear()  
    st.session_state.clear()  
    st.experimental_rerun() 
    stage1()


if st.button("Reset"):
    reset_cache() 
    stage1()
    