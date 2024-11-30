import streamlit as st
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
from sklearn.naive_bayes import GaussianNB  # Import GaussianNB
import os
st.markdown("""
            <style>







               </style>
        """, unsafe_allow_html=True)  
# Stage 1: Load dataset, train models, and show accuracy results
@st.cache_resource
def load_and_train_models():
    # Load dataset
    csv_file_path = "C:/Users/Admin/Documents/ITE105/Lab3/heart_failure_clinical_records_dataset.csv"
    try:
        dataframe = pd.read_csv(csv_file_path)
        st.dataframe(dataframe)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Split features and target variable
    X = dataframe.drop('DEATH_EVENT', axis=1)
    y = dataframe['DEATH_EVENT']

    # Train-test split parameters
    test_size = 0.2
    random_seed = 42
    
    
    # Initialize accuracies list
    accuracies = []

    # Decision Tree model
    
    test_size = 0.2
    random_seed = 42
    max_depth = 20
    min_samples_split = 10
    min_samples_leaf = 10
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)


    model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,  # Use min_samples_split parameter
    min_samples_leaf=min_samples_leaf,    # Use min_samples_leaf parameter
    random_state=random_seed
        )

        # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "DC", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    



    #GS
    var_smoothing=-9
    test_size=0.2
    random_seed=42
        

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    var_smoothing_value = 10 ** var_smoothing

    model = GaussianNB(var_smoothing=var_smoothing_value)
    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "GS", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
   

    #gradient
    test_size = 0.3
    random_seed = 42
    n_estimators = 10
    learning_rate = 0.2
    max_depth = 2

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_seed
    )



    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "Gradient Boosting", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    






    #near
    test_size =0.2
    random_seed =42
    n_neighbors = 25
    metric = "euclidean"

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "KNN", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    # Display the table in Streamlit

   
    #st.title("Logistic Regression Classifier with LOOCV")


    ##LGRC
    test_size = 0.2
    random_seed = 42
    solver = "liblinear"
    penalty = "l1"
    C = 0.20

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    #st.header("Leave-One-Out Cross-Validation (LOOCV)")
    model = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=1000)
    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "LG", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
   

    # Display the table in Streamlit


    #st.title("Multi-Layer Perceptron Classifier with LOOCV")

    hidden_layer_sizes = 25
    learning_rate = "adaptive"
    activation = "logistic"
    max_iter = 100

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), learning_rate=learning_rate, 
                            activation=activation, max_iter=max_iter)

    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "MLP", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    
    # Display the table in Streamlit

 
    #st.title("Random Forest Classifier with LOOCV")

    n_estimators = 50
    max_depth = 50
    random_state = 42
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "RFT", "Mean Accuracy": f"{accuracy * 100:.2f}%"})


    # Display the table in Streamli

  

    #st.title("Perceptron Classifier with LOOCV")
    max_iter = 100
    random_state = 42

    # Perceptron model
    model = Perceptron(max_iter=max_iter, random_state=random_state)
    laccuracies = []
    log_losses = []
    y_true = []
    y_probs = []

    for train_index, test_index in loocv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loocv_accuracies.append(accuracy_score([y_test], [y_pred]))
        y_true.append(y_test)
        try:
            y_probs.append(model.decision_function(X_test)[0])  # For log loss and AUC
        except AttributeError:
            y_probs.append(0)  # Handle cases where decision function isn't supported

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "Perceptron", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    
    # Display the table in Streamlit


    

    
    #st.title("SVM Classifier with LOOCV")
    kernel = "sigmoid"
    C = 0.1
    random_state =42

    # SVM model
    model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies.append({"Model": "svm", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    

    # Display the accuracies and button to proceed

    return accuracies
@st.cache_resource  
def load_and_train_models2():
    # Load dataset
    csv_file_path = "C:/Users/Admin/Documents/ITE105/Lab3/heart_failure_clinical_records_dataset.csv"
    try:
        dataframe = pd.read_csv(csv_file_path)
        
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Split features and target variable
    X = dataframe.drop('DEATH_EVENT', axis=1)
    y = dataframe['DEATH_EVENT']

    # Train-test split parameters
    test_size = 0.2
    random_seed = 42
    
    
    # Initialize accuracies list
    accuracies2 = []

    # Decision Tree model
    
    test_size = 0.2
    random_seed = 42
    max_depth = 20
    min_samples_split = 10
    min_samples_leaf = 10
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)


    model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,  # Use min_samples_split parameter
    min_samples_leaf=min_samples_leaf,    # Use min_samples_leaf parameter
    random_state=random_seed
        )

        # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "DC", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    



    #GS
    var_smoothing=-9
    test_size=0.2
    random_seed=42
        

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    var_smoothing_value = 10 ** var_smoothing

    model = GaussianNB(var_smoothing=var_smoothing_value)
    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "GS", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
   
    #gradient
    test_size = 0.3
    random_seed = 42
    n_estimators = 10
    learning_rate = 0.2
    max_depth = 2

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_seed
    )



    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "Gradient Boosting", "Mean Accuracy": f"{accuracy * 100:.2f}%"})



    #near
    test_size =0.2
    random_seed =42
    n_neighbors = 25
    metric = "euclidean"

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "KNN", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    # Display the table in Streamlit

   
    #st.title("Logistic Regression Classifier with LOOCV")


    ##LGRC
    test_size = 0.2
    random_seed = 42
    solver = "liblinear"
    penalty = "l1"
    C = 0.20

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    #st.header("Leave-One-Out Cross-Validation (LOOCV)")
    model = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=1000)
    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "LG", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
   

    # Display the table in Streamlit


    #st.title("Multi-Layer Perceptron Classifier with LOOCV")

    hidden_layer_sizes = 25
    learning_rate = "adaptive"
    activation = "logistic"
    max_iter = 100

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), learning_rate=learning_rate, 
                            activation=activation, max_iter=max_iter)

    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "MLP", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    
    # Display the table in Streamlit

 
    #st.title("Random Forest Classifier with LOOCV")

    n_estimators = 50
    max_depth = 50
    random_state = 42
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "RFT", "Mean Accuracy": f"{accuracy * 100:.2f}%"})


    # Display the table in Streamli

  

    #st.title("Perceptron Classifier with LOOCV")
    max_iter = 100
    random_state = 42

    # Perceptron model
    model = Perceptron(max_iter=max_iter, random_state=random_state)
    laccuracies = []
    log_losses = []
    y_true = []
    y_probs = []

    for train_index, test_index in loocv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loocv_accuracies.append(accuracy_score([y_test], [y_pred]))
        y_true.append(y_test)
        try:
            y_probs.append(model.decision_function(X_test)[0])  # For log loss and AUC
        except AttributeError:
            y_probs.append(0)  # Handle cases where decision function isn't supported

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "Perceptron", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    
    # Display the table in Streamlit


    

    
    #st.title("SVM Classifier with LOOCV")
    kernel = "sigmoid"
    C = 0.1
    random_state =42

    # SVM model
    model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    accuracies2.append({"Model": "svm", "Mean Accuracy": f"{accuracy * 100:.2f}%"})
    

    # Display the accuracies and button to proceed

    return accuracies2
# Stage 1: Load dataset, train models, and show accuracy results

def stage1():
    
    st.title("Heart Failure Model Training ")

    st.success("Dataset loaded successfully!")
    # Load data and train models using cached function
    # Load data and train models using cached function
    accuracies = load_and_train_models()

    # Display the dataframe only once here in Stage 1
    
    

    # Display the accuracies in a dataframe format
    accuracy_df = pd.DataFrame(accuracies)
    st.write("Accuracy Results:", accuracy_df)

    
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
    accuracies_numeric = [float(accuracy["Mean Accuracy"].rstrip('%')) for accuracy in accuracies2]

    fig, ax = plt.subplots()
    ax.bar(models, accuracies_numeric, color='skyblue', width=0.5)
    ax.set_title("Model Mean Accuracies", color='white')
    ax.set_xlabel("Model", color='white')
    ax.set_ylabel("Mean Accuracy (%)", color='white')
    ax.set_ylim(0, 100)
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

# Stage 3: Model selection and hyperparameter tuning
def stage3():
    st.title("Model Hyper Parameter Tuning")
    # Load model accuracies
    accuracies2 = load_and_train_models2()

    if "accuracies2" not in st.session_state:
        st.error("No accuracy data found. Please go back to Stage 1.")
        return  # Exit if accuracies are not available+
    
    # Get the stored accuracies from session state
    accuracies2 = st.session_state.accuracies2
    
    # Extract numeric accuracies and find the highest one
    accuracies_numeric = [float(accuracy["Mean Accuracy"].rstrip('%')) for accuracy in accuracies2]
    highest_accuracy = max(accuracies_numeric)
    highest_model_index = accuracies_numeric.index(highest_accuracy)
    
    # Load dataset
    csv_file_path = "C:/Users/Admin/Documents/ITE105/Lab3/heart_failure_clinical_records_dataset.csv"
    try:
        dataframe = pd.read_csv(csv_file_path)
        st.success("Dataset loaded successfully!")
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return
    
    # Split features and target variable
    X = dataframe.drop('DEATH_EVENT', axis=1)
    y = dataframe['DEATH_EVENT']
    
    # Model selection based on highest accuracy
    if highest_model_index == 0:  # Decision Tree
        st.session_state.selected_model = "dt"
        dt_tuning(X, y)
    elif highest_model_index == 1:  # Gaussian NB
        st.session_state.selected_model = "ds"
        gaussian_nb_tuning(X, y)
    elif highest_model_index == 2:  # Gaussian NB
        st.session_state.selected_model = "gbm"
        gbm_tuning(X, y)
    
    elif highest_model_index == 3:  # Gaussian NB
        st.session_state.selected_model = "knn"
        knn_tuning(X, y)#KNN
    elif highest_model_index == 4:  # Gaussian NB
        st.session_state.selected_model = "knn"
        lg_tuning(X, y)
    elif highest_model_index == 5:  # Gaussian NB
        st.session_state.selected_model = "mlp"
        mlp_tuning(X, y)
    elif highest_model_index == 6:  # Gaussian NB
        st.session_state.selected_model = "rft"
        rft_tuning(X, y)
    elif highest_model_index == 7:  # Gaussian NB
        st.session_state.selected_model = "perceptron"
        perceptron_tuning(X, y)  
    else:  # Gradient Boosting
        st.session_state.selected_model = "svm"
        svm_tuning(X, y)

# Decision Tree Hyperparameter Tuning
def dt_tuning(X, y):
    st.subheader("Decision Tree Tuning")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2,key="test5")
    random_seed = st.slider('Random Seed', 1, 100, 42, 1,value=42)
    max_depth = st.slider('Max Depth', 1, 50, 20, 1,value=20)
    min_samples_split = st.slider('Min Samples Split', 2, 20, 10, 1,value=10)
    min_samples_leaf = st.slider('Min Samples Leaf', 1, 20, 10, 1,value=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, 
                                   min_samples_leaf=min_samples_leaf, random_state=random_seed)
    
    perform_loocv(X, y, model, "dt")

# Gaussian Naive Bayes Hyperparameter Tuning
def gaussian_nb_tuning(X, y):
    st.subheader("Gaussian NB Tuning")
    var_smoothing = st.number_input('Var Smoothing (log-scale)', -15, -1, -9, 1,value = -9)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2,key="test5")
    random_seed = st.slider('Random Seed', 1, 100, 42, 1,value=42)

    var_smoothing_value = 10 ** var_smoothing
    model = GaussianNB(var_smoothing=var_smoothing_value)

    perform_loocv(X, y, model, "gs")

# Gradient Boosting Hyperparameter Tuning
def gbm_tuning(X,y):
    #goal is ma labwan ni82.61%
    #current highest sa gbm is random seed 42,n estimators 10,lr 0.2 , md = 2
    test_size = 0.2
    random_seed = st.slider("Random seed",min_value=1,max_value=1000,value=42) 
    n_estimators =  st.slider("N Estimators",min_value=1,max_value=1000,value=100) 
    learning_rate =  st.number_input("Learning Rate",min_value=0.005,max_value=1.00,value=0.2) 
    max_depth =  st.slider("Max Depth",min_value=1,max_value=100,value=10) 
    #result atm 11/30 2:35 is 79.26
    # Train-test split
    
    
  
    # Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_seed
    )
    perform_loocv(X, y, model, "gbm")



    
def knn_tuning(X, y):
    st.subheader("knn")
    
    n_neighbors = st.slider("Number of Neighbors (k)", min_value=1, max_value=50, value=25, step=1)
    metric = st.selectbox("Distance Metric", options=["euclidean", "manhattan", "chebyshev", "minkowski"], index=0)

  
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    perform_loocv(X, y, model, "knn")
def lg_tuning(X, y):
    st.subheader("LG")
    
    solver = st.selectbox("Solver", options=["liblinear", "lbfgs", "saga", "newton-cg"], index=0)
    penalty = st.selectbox("Penalty", options=["l1", "l2", "elasticnet", "none"], index=0)
    C = st.slider("Inverse Regularization Strength (C)", min_value=0.01, max_value=10.0, value=0.20, step=0.01)

  
    model = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=1000)

    perform_loocv(X, y, model, "lg")

def mlp_tuning(X, y):
    st.subheader("MLP")
    
    
    hidden_layer_sizes = st.slider(
    "Hidden Layer Sizes",
    min_value=1,
    max_value=100,
    value=25,  # Default value
    step=1
    )

    # Slider for `max_iter`
    max_iter = st.slider(
        "Maximum Iterations",
        min_value=100,
        max_value=1000,
        value=100,  # Default value
        step=50
    )

    # Slider for `learning_rate` (mapped to options via indices)
    learning_rate_options = ["constant", "invscaling", "adaptive"]
    learning_rate_index = st.slider(
        "Learning Rate (Index)",
        min_value=0,
        max_value=2,
        value=2,  # Default: "adaptive"
        step=1
    )
    learning_rate = learning_rate_options[learning_rate_index]

    # Slider for `activation` (mapped to options via indices)
    activation_options = ["identity", "logistic", "tanh", "relu"]
    activation_index = st.slider(
        "Activation Function (Index)",
        min_value=0,
        max_value=3,
        value=1,  # Default: "logistic"
        step=1
    )
    activation = activation_options[activation_index]
  
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), learning_rate=learning_rate, 
                        activation=activation, max_iter=max_iter)

    perform_loocv(X, y, model, "mlp")

def rft_tuning(X, y):
    st.subheader("RFT")
    
    n_estimators = st.slider("Number of Estimators", 10, 200, 100,value=50)
    max_depth = st.slider("Maximum Depth", 1, 50, 10,value=50)
    random_state = st.slider("Random Seed", 0, 100, 42,value=42)

  
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    perform_loocv(X, y, model, "rft")

def perceptron_tuning(X, y):
    st.subheader("Perceptron")
    
    max_iter = st.slider("Max Iterations", 100, 1000, 200 ,value=100)
    random_state = st.slider("Random Seed", 0, 100, 42,value=42)

  
    model = Perceptron(max_iter=max_iter, random_state=random_state)
    perform_loocv(X, y, model, "perceptron")
def svm_tuning(X, y):
    st.subheader("SVM")
    
    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=3)
    C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
    random_state = st.slider("Random Seed", 0, 100, 42)

  
    model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)
    perform_loocv(X, y, model, "svm")


# Perform LOOCV and display results
def perform_loocv(X, y, model, model_name):
    test_size=0.2
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)
    
        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)
    st.write(f"The model with the highest accuracy is : {model_name}, with a mean Accuracy: {accuracy * 100:.2f}%")

    model_filename = "classificationmodel.joblib"
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
    if st.button("Go to Heart Failure Prediction Application"):
        st.session_state.stage = 4  # Set stage to 2
        st.rerun()  # Trigger a rerun to go to Stage 2
def stage4():
    # Load model accuracies
    
    st.title("Heart Failure Prediction Application")
    # Extract numeric accuracies and find the highest one
    # Step 1: Upload the pre-trained model
    csv_file_path2 = "C:/Users/Admin/Documents/ITE105/LabFinal/classificationmodel.joblib"
    try:
        model = joblib.load(csv_file_path2)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Input form for user data
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase Level", min_value=0)
    diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100)
    high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])
    platelets = st.number_input("Platelets Count", min_value=0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0.0)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
    time = st.number_input("Time", min_value=0)  # Ensure you include this if it's a feature

    # Create a button to predict
    if st.button("Predict"):
        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            "age": [age],
            "anaemia": [anaemia],
            "creatinine_phosphokinase": [creatinine_phosphokinase],
            "diabetes": [diabetes],
            "ejection_fraction": [ejection_fraction],
            "high_blood_pressure": [high_blood_pressure],
            "platelets": [platelets],
            "serum_creatinine": [serum_creatinine],
            "serum_sodium": [serum_sodium],
            "sex": [sex],
            "smoking": [smoking],
            "time": [time]  # Include 'time' if it's used in the model
        })

        # Make predictions
        prediction = model.predict(input_data)
        predicted_probabilities = model.predict_proba(input_data)

        # Display results
        if prediction[0] == 1:
            st.write("Prediction: The patient is likely to have heart failure.")
        else:
            st.write("Prediction: The patient is unlikely to have heart failure.")

        st.write("Probability of Heart Failure:", predicted_probabilities)


    if st.button("Go Back to Model Comparison"):
        st.session_state.stage = 3  # Set stage to 1
        st.rerun()  # Trigger a rerun to go back to Stage 1

    # Button to proceed to further stages
     # Button to proceed to Stage 2
    if st.button("Go to Heart Failure Prediction Application"):
        st.session_state.stage = 1  # Set stage to 2
        st.rerun()  # Trigger a rerun to go to Stage 2
# Ensure session state tracking
if "stage" not in st.session_state:
    st.session_state.stage = 1  # Default to Stage 1

if "accuracies" not in st.session_state:
    st.session_state.accuracies = load_and_train_models()  # Load and cache the accuracies if not already done

if "accuracies2" not in st.session_state:
    st.session_state.accuracies2 = load_and_train_models2()  # Load and cache the accuracies if not already done

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
    