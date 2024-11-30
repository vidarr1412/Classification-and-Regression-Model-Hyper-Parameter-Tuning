
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
import joblib

def load_train_models():
# Load dataset
    csv_file_path = "C:/Users/Admin/Documents/ITE105/LabFinal/sealevel.csv"
    try:
        data = pd.read_csv(csv_file_path)
        # Display the dataset preview
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
            st.subheader("Regression Model")
            accuracies=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, train_size=0.8)

                # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

 
            accuracies.append({"Model": "Linear Regression", "MAE": f"{mae:.2f}"})
            ada_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)

            # Train the model
            ada_regressor.fit(X_train, y_train)

            # Predict on the test set
            y_pred = ada_regressor.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
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
            mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)  # Adjust hidden layers and max_iter
            mlp_regressor.fit(X_train, y_train)
            y_pred = mlp_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "MLP Regressor", "MAE": f"{mae:.2f}"})
            rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune the number of trees (n_estimators)
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Random Forest Regressor", "MAE": f"{mae:.2f}"})

            # Support Vector Machine (SVM) Regressor model
            svm_regressor = SVR(kernel='rbf', C=100, epsilon=0.1)  # Adjust C, epsilon, and kernel type for tuning
            svm_regressor.fit(X_train, y_train)
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "SVM Regressor", "MAE": f"{mae:.2f}"})
            elastic_net_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust alpha and l1_ratio parameters
            elastic_net_regressor.fit(X_train, y_train)
            y_pred = elastic_net_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Elastic Net", "MAE": f"{mae:.2f}"})
            decision_tree_regressor = DecisionTreeRegressor(random_state=42)  # You can tune max_depth, min_samples_split, etc.
            decision_tree_regressor.fit(X_train, y_train)
            y_pred = decision_tree_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "Decision Tree", "MAE": f"{mae:.2f}"})
            knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can tune the number of neighbors
            knn_regressor.fit(X_train, y_train)
            y_pred = knn_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies.append({"Model": "KNN Regressor", "MAE": f"{mae:.2f}"})
    
            return accuracies
    
def load_train_models2():
# Load dataset
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

            accuracies2=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, train_size=0.8)

                # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

 
            accuracies2.append({"Model": "Linear Regression", "MAE": f"{mae:.2f}"})
            ada_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)

            # Train the model
            ada_regressor.fit(X_train, y_train)

            # Predict on the test set
            y_pred = ada_regressor.predict(X_test)

            # Evaluate the model
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
            mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)  # Adjust hidden layers and max_iter
            mlp_regressor.fit(X_train, y_train)
            y_pred = mlp_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "MLP Regressor", "MAE": f"{mae:.2f}"})


            rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune the number of trees (n_estimators)
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Random Forest Regressor", "MAE": f"{mae:.2f}"})

            # Support Vector Machine (SVM) Regressor model
            svm_regressor = SVR(kernel='rbf', C=100, epsilon=0.1)  # Adjust C, epsilon, and kernel type for tuning
            svm_regressor.fit(X_train, y_train)
            y_pred = svm_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "SVM Regressor", "MAE": f"{mae:.2f}"})
            elastic_net_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust alpha and l1_ratio parameters
            elastic_net_regressor.fit(X_train, y_train)
            y_pred = elastic_net_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Elastic Net", "MAE": f"{mae:.2f}"})
            decision_tree_regressor = DecisionTreeRegressor(random_state=42)  # You can tune max_depth, min_samples_split, etc.
            decision_tree_regressor.fit(X_train, y_train)
            y_pred = decision_tree_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "Decision Tree", "MAE": f"{mae:.2f}"})
            knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can tune the number of neighbors
            knn_regressor.fit(X_train, y_train)
            y_pred = knn_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            accuracies2.append({"Model": "KNN Regressor", "MAE": f"{mae:.2f}"})
            
            return accuracies2

def stage1():
    
    st.title("Heart Failure Model Training ")

    st.success("Dataset loaded successfully!")
    # Load data and train models using cached function
    # Load data and train models using cached function
    accuracies = load_train_models()

    # Display the dataframe only once here in Stage 1
    
    

    # Display the accuracies in a dataframe format
    accuracy_df = pd.DataFrame(accuracies)
    st.write("MAE results:", accuracy_df)

    
    # Button to proceed to Stage 2
    if st.button("Proceed to Model Accuracy Comparison"):
        st.session_state.stage = 2  # Set stage to 2
        st.rerun()  # Trigger a rerun to go to Stage 2
# Stage 2: Show the accuracy comparison as a bar chart
def stage2():
    st.title("Model Accuracy Comparison")

    # Ensure that accuracies are available from session state
    if "accuracies2" not in st.session_state:
        st.error("No accuracy data found. Please go back to Stage 1.")
        return  # Exit if accuracies are not available+
    
    # Get the stored accuracies from session state
    accuracies2 = st.session_state.accuracies2

    # Create a bar graph
    models = [accuracy["Model"] for accuracy in accuracies2]
    accuracies_numeric = [float(accuracy["MAE"].rstrip('%')) for accuracy in accuracies2]

    fig, ax = plt.subplots()
    ax.bar(models, accuracies_numeric, color='skyblue', width=0.5)
    ax.set_title("MAE", color='white')
    ax.set_xlabel("Model", color='white')
    ax.set_ylabel("Mean Accuracy (%)", color='white')
    ax.set_ylim(0, 15)
    plt.xticks(rotation=25, color='white')
    plt.yticks(color='white')

    # Set the color of the axis lines to white
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Make the background transparent
    fig.patch.set_facecolor('none')
    ax.patch.set_facecolor('none')

    # Display the bar graph in Streamlit
    st.pyplot(fig)

    # Button to go back to Stage 1
    if st.button("Go Back to Page 1"):
        st.session_state.stage = 1  # Set stage to 1
        st.rerun()  # Trigger a rerun to go back to Stage 1

    # Button to proceed to further stages
     # Button to proceed to Stage 2
    if st.button("Proceed to Hyperparameter Tuning"):
        st.session_state.stage = 3  # Set stage to 2
        st.rerun()  # Trigger a rerun to go to Stage 2
def stage3():
    st.title("Model Hyper Parameter Tuning")
    # Load model accuracies
    accuracies2 = load_train_models2()

    if "accuracies2" not in st.session_state:
        st.error("No accuracy data found. Please go back to Stage 1.")
        return  # Exit if accuracies are not available+
    
    # Get the stored accuracies from session state
    accuracies2 = st.session_state.accuracies2
    
    # Extract numeric accuracies and find the highest one
    accuracies_numeric = [float(accuracy["MAE"].rstrip('%')) for accuracy in accuracies2]

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
            fit_intercept = st.sidebar.checkbox("Fit Intercept", True)
            model = LinearRegression(fit_intercept=fit_intercept)

            
            

    
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

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
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

        
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
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

            
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
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
            # Select features and target variable
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

         
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
    perform_train_split(X, y, model, "mlp")

#Random Forest Regressor
def rft_tuning(X, y):
    st.subheader("rft")
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

            
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=100, step=10)
            train_size = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8)
            random_seed = st.number_input("Random Seed",min_value=10,max_value=500,value=123)
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
            st.write(f"The model with the highest accuracy is : rft, with a mean Accuracy: {mae:.2f}%")
                
                    

    
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

            
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
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
            X = data[["TotalWeightedObservations", "GMSL_noGIA", "StdDevGMSL_noGIA", "SmoothedGSML_noGIA",
                    "GMSL_GIA", "StdDevGMSL_GIA", "SmoothedGSML_GIA"]]
            y = data["SmoothedGSML_GIA_sigremoved"]

            
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
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

            
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
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

            
            n_estimators = st.sidebar.slider("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            loss = st.sidebar.selectbox("Loss Function", options=["linear", "square", "exponential"])

            # Initialize and train AdaBoost Regressor
            model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss,
                random_state=123
            )
            
            

    
    perform_train_split(X, y, model, "knn")
def perform_train_split(X, y, model, model_name):
    

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
    if st.button("Go Back to Model Comparison"):
        st.session_state.stage = 2  # Set stage to 1
        st.rerun()  # Trigger a rerun to go back to Stage 1

    # Button to proceed to further stages
     # Button to proceed to Stage 2
    if st.button("Go Predict"):
        st.session_state.stage = 4  # Set stage to 2
        st.rerun()  # Trigger a rerun to go to Stage 2
# Gaussian Naive Bayes Hyperparameter Tuning
def stage4():
    # Load model accuracies
    
    st.title("Sea Level Prediction Application")
    # Extract numeric accuracies and find the highest one
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
    year = st.number_input("Year", value=2023, step=1, format="%d")
    TotalWeightedObservations = st.number_input("Total Weighted Observations", value=0.0, format="%.2f")
    GMSL_noGIA = st.number_input("GMSL (Global Isostatic Adjustment (GIA) not applied) variation (mm) with respect to 20-year TOPEX/Jason collinear mean reference:", value=0.0, format="%.2f")
    StdDevGMSL_noGIA = st.number_input("Standard Deviation of GMSL (GIA not applied) variation estimate (mm)", value=0.0, format="%.2f")
    SmoothedGSML_noGIA = st.number_input("Smoothed (60-day Gaussian type filter) GMSL (GIA not applied) variation (mm)", value=0.0, format="%.2f")
    GMSL_GIA = st.number_input("GMSL (Global Isostatic Adjustment (GIA) applied) variation (mm) with respect to 20-year TOPEX/Jason collinear mean reference:", value=0.0, format="%.2f")
    StdDevGMSL_GIA = st.number_input("Standard Deviation of GMSL (GIA applied) variation estimate (mm)", value=0.0, format="%.2f")
    SmoothedGSML_GIA = st.number_input("Smoothed (60-day Gaussian type filter) GMSL (GIA applied) variation (mm)", value=0.0, format="%.2f")

    # Organize inputs into a single feature array
    input_features = np.array([[TotalWeightedObservations, GMSL_noGIA, StdDevGMSL_noGIA,
                                SmoothedGSML_noGIA, GMSL_GIA, StdDevGMSL_GIA, SmoothedGSML_GIA]])

    # Step 3: Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_features)[0]  # Get the prediction
        st.subheader("Prediction Result")
        st.write(f"The Predicted Sea Rise Level by  {int(year):d} is  {prediction:.2f} in millimeters")

    

    if st.button("Go Back to Hyper Parameter Tuning"):
        st.session_state.stage = 3  # Set stage to 1
        st.rerun()  # Trigger a rerun to go back to Stage 1

    # Button to proceed to further stages
     # Button to proceed to Stage 2
    if st.button("Go to Back from the Start"):
        st.session_state.stage = 1  # Set stage to 2
        st.rerun()  # Trigger a rerun to go to Stage 2
# Ensure session state tracking


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


if st.button("Reset Cache"):
    reset_cache()  # Reset the cache when the button is clicked
    stage1()
    