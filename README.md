
# Machine Learning Model Selection

A user-friendly web application developed as part of the CS351 - Artificial Intelligence course at GIKI. 

This project simplifies the process of selecting and evaluating machine learning models, focusing on both regression and classification tasks, enabling users to preprocess data, experiment with models, and make informed decisions. Selecting the most suitable machine learning model for a dataset can be challenging, especially for individuals with limited technical expertise. This project addresses this challenge by providing an intuitive, no-code platform for preprocessing datasets, selecting models, and evaluating their performance. 

## Deployment

Deployed at: [https://mlmodelselection.streamlit.app/](https://mlmodelselection.streamlit.app/)

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/JunaidSalim/ML_Model_Selection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML_Model_Selection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   # Without CatBoost Model
   streamlit run app.py 

   # With CatBoost Model
   streamlit run main.py 
   ```

## Functionality

### 1. Upload Dataset
Easily upload datasets in `.csv` format. The interface ensures the file is ready for processing.
![Dataset Upload](https://github.com/user-attachments/assets/2cad8d29-8d00-4fab-86cf-ea5e1ac07654)


### 2. Dataset Preview
Preview the uploaded dataset, including statistical summaries of numerical features.

![4](https://github.com/user-attachments/assets/3a72d932-b2e9-4e09-ac70-929b8c216d38)
![2](https://github.com/user-attachments/assets/fb9bc6c9-7edd-45db-a0ef-efe60bfcde90)
![3](https://github.com/user-attachments/assets/b9d667e4-1d10-4d56-a415-301fc8fed6f3)
![5](https://github.com/user-attachments/assets/97833178-b6ea-441a-91f2-89564fb51231)

### 3. Data Preprocessing
- Handle missing values.
- Normalize data.
- Apply one-hot encoding for categorical variables.
- Select features and target variables for modeling.
![6](https://github.com/user-attachments/assets/d71a8762-21ae-4c12-aaf4-bae1420a8263)

### 4. Task Selection
The app detects the task type (regression or classification) and allows manual adjustments if necessary.
![Task Selection](https://github.com/user-attachments/assets/37d03b3b-f121-49f3-adad-9be0473340c6)


### 5. Hyperparameter Tuning
Optimize model performance by adjusting parameters interactively.

#### Hyperparameters for Regression
![Hyperparameter Tuning](https://github.com/user-attachments/assets/09eaf972-0dfd-4f80-af2e-91d00b606c2a)

#### Hyperparameters for Classification
![9](https://github.com/user-attachments/assets/426ecf78-a5ca-4811-8f6e-0c21bbdae443)


### 6. Results and Model Comparison
View and compare evaluation metrics for various models to determine the best fit for your dataset.
![Model Results](https://github.com/user-attachments/assets/5ca3978a-9cce-4ebc-8426-c9cfaf33c5b6)



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
