
# Machine Learning Model Selection

A user-friendly web application developed as part of the CS351 - Artificial Intelligence course at GIKI. 

This project simplifies the process of selecting and evaluating machine learning models, focusing on both regression 
and classification tasks, enabling users to preprocess data, experiment with models, and make informed decisions. Selecting the most suitable machine learning model for a dataset can be challenging, especially for individuals with limited technical expertise. This project addresses this challenge by providing an intuitive, no-code platform for preprocessing datasets, selecting models, 
and evaluating their performance. It bridges the gap between raw data and actionable insights.

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
   streamlit run app.py
   ```

## Functionality

### 1. Upload Dataset
Easily upload datasets in `.csv` format. The interface ensures the file is ready for processing.
![Dataset Upload](path/to/image)

### 2. Dataset Preview
Preview the uploaded dataset, including statistical summaries of numerical features.
![Dataset Preview](path/to/image)

### 3. Data Preprocessing
- Handle missing values.
- Normalize data.
- Apply one-hot encoding for categorical variables.
- Select features and target variables for modeling.
![Preprocessing Options](path/to/image)

### 4. Task Selection
The app detects the task type (regression or classification) and allows manual adjustments if necessary.
![Task Selection](path/to/image)

### 5. Hyperparameter Tuning
Optimize model performance by adjusting parameters interactively.
![Hyperparameter Tuning](path/to/image)

### 6. Results and Model Comparison
View and compare evaluation metrics for various models to determine the best fit for your dataset.
![Model Results](path/to/image)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
