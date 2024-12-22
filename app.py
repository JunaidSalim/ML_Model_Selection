import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt

st.title("Machine Learning Web App")
st.subheader("Reg No: 2022243 | Name: Junaid Saleem")

uploadedFile = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploadedFile:
    data = pd.read_csv(uploadedFile)
    st.write("Dataset Preview:")
    st.write(data.head())

    st.write("Feature Distribution:")
    numericColumns = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numericColumns) == 0:
        st.write("No numeric columns available for visualization.")
    else:
        columnCount = 0
        while columnCount < len(numericColumns):
            remainingColumns = len(numericColumns) - columnCount
            numCols = min(4, remainingColumns)
            
            fig, axs = plt.subplots(nrows=1, ncols=numCols, figsize=(15, 3))
            
            if numCols == 1:
                axs = [axs]
            
            for i in range(numCols):
                col = numericColumns[columnCount]
                axs[i].hist(data[col], bins=20, color='blue', edgecolor='black')
                axs[i].set_title(f"{col}")
                columnCount += 1
            
            st.pyplot(fig)
            plt.close(fig)

    st.write("Remove Columns:")
    columnsToRemove = st.multiselect("Select columns to remove", data.columns)
    if columnsToRemove:
        data = data.drop(columns=columnsToRemove)

    targetColumn = st.selectbox("Select the target column", data.columns, index=len(data.columns) - 1)

    st.write("Handle Missing Values:")
    missingMethod = st.radio("Select method:", ['Mean', 'Median', 'Mode', 'Remove'])
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if missingMethod == 'Mean':
                data[column].fillna(data[column].mean(), inplace=True)
            elif missingMethod == 'Median':
                data[column].fillna(data[column].median(), inplace=True)
            elif missingMethod == 'Mode':
                data[column].fillna(data[column].mode()[0], inplace=True)
            elif missingMethod == 'Remove':
                data = data.dropna(subset=[column])

    st.write("Data Preprocessing Options:")
    normalizeData = st.checkbox("Normalize Data")
    oneHotEncode = st.checkbox("One-Hot Encode Categorical Features")

    inputFeatures = data.drop(columns=[targetColumn])
    targetFeature = data[targetColumn]

    if targetFeature.dtype == 'object':
        targetFeature = LabelEncoder().fit_transform(targetFeature)

    if targetFeature.nunique() == 2:
        uniqueClasses = targetFeature.unique()
        targetFeature = targetFeature.map({uniqueClasses[0]: 0, uniqueClasses[1]: 1})

    if oneHotEncode:
        columnsToEncode = st.multiselect("Select columns to one-hot encode", inputFeatures.columns)
        inputFeatures = pd.get_dummies(inputFeatures, columns=columnsToEncode)

    if normalizeData:
        scaler = StandardScaler()
        inputFeatures = pd.DataFrame(scaler.fit_transform(inputFeatures), columns=inputFeatures.columns)

    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
        inputFeatures, targetFeature, test_size=0.2, random_state=42
    )

    defaultTask = "Classification" if targetFeature.nunique() < 10 else "Regression"
    taskType = st.selectbox("Choose a task (Default detected: {})".format(defaultTask), ["Regression", "Classification"], index=0 if defaultTask == "Regression" else 1)

    hyperCols = st.columns(3)

    if taskType == "Regression":
        svrC = hyperCols[0].number_input("SVR: C", 0.01, 10.0, 1.0)
        svrKernel = hyperCols[1].selectbox("SVR: Kernel", ["linear", "poly", "rbf", "sigmoid"])
        dtMaxDepth = hyperCols[2].number_input("Decision Tree: Max Depth", 1, 50, 5)
        rfNEstimators = hyperCols[0].number_input("Random Forest: Estimators", 10, 500, 100)
        xgbLearningRate = hyperCols[1].number_input("XGBoost: Learning Rate", 0.01, 1.0, 0.1)
        xgbNEstimators = hyperCols[2].number_input("XGBoost: Estimators", 10, 500, 100)
        catboostIterations = hyperCols[0].number_input("CatBoost: Iterations", 10, 1000, 100)

        models = {
            "Multiple Linear Regression": LinearRegression(),
            "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
            "SVR": SVR(C=svrC, kernel=svrKernel),
            "Decision Tree Regression": DecisionTreeRegressor(max_depth=dtMaxDepth),
            "Random Forest Regression": RandomForestRegressor(n_estimators=rfNEstimators),
            "XGBoost Regression": xgb.XGBRegressor(learning_rate=xgbLearningRate, n_estimators=xgbNEstimators),
            "CatBoost Regression": CatBoostRegressor(iterations=catboostIterations, verbose=0)
        }

        results = []
        for modelName, model in models.items():
            model.fit(trainFeatures, trainLabels)
            predictions = model.predict(testFeatures)

            mse = mean_squared_error(testLabels, predictions)
            r2 = r2_score(testLabels, predictions)

            results.append({
                "Model": modelName,
                "Mean Squared Error": mse,
                "R² Score": r2
            })

        resultsDf = pd.DataFrame(results)
        st.write("Regression Results:")
        st.write(resultsDf)

    elif taskType == "Classification":
        knnNeighbors = hyperCols[0].number_input("KNN: Neighbors", 1, 50, 5)
        svcC = hyperCols[1].number_input("SVC: C", 0.01, 10.0, 1.0)
        svcKernel = hyperCols[2].selectbox("SVC: Kernel", ["linear", "poly", "rbf", "sigmoid"])
        dtMaxDepth = hyperCols[0].number_input("Decision Tree: Max Depth", 1, 50, 5)
        rfNEstimators = hyperCols[1].number_input("Random Forest: Estimators", 10, 500, 100)
        xgbLearningRate = hyperCols[2].number_input("XGBoost: Learning Rate", 0.01, 1.0, 0.1)
        xgbNEstimators = hyperCols[0].number_input("XGBoost: Estimators", 10, 500, 100)
        catboostIterations = hyperCols[1].number_input("CatBoost: Iterations", 10, 1000, 100)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=knnNeighbors),
            "SVC": SVC(C=svcC, kernel=svcKernel),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=dtMaxDepth),
            "Random Forest": RandomForestClassifier(n_estimators=rfNEstimators),
            "XGBoost": xgb.XGBClassifier(learning_rate=xgbLearningRate, n_estimators=xgbNEstimators),
            "CatBoost": CatBoostClassifier(iterations=catboostIterations, verbose=0)
        }

        if targetFeature.nunique() > 2:
            models = {key: model for key, model in models.items() if not isinstance(model, (LogisticRegression, SVC, xgb.XGBClassifier, CatBoostClassifier))}

        results = []
        for modelName, model in models.items():
            model.fit(trainFeatures, trainLabels)
            predictions = model.predict(testFeatures)

            accuracy = accuracy_score(testLabels, predictions)
            precision = precision_score(testLabels, predictions, average='weighted')
            recall = recall_score(testLabels, predictions, average='weighted')
            f1 = f1_score(testLabels, predictions, average='weighted')

            results.append({
                "Model": modelName,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

        resultsDf = pd.DataFrame(results)
        st.write("Classification Results:")
        st.write(resultsDf)
