# Simple Linear Regression Project

This project demonstrates the implementation of a Simple Linear Regression model using Python. The goal is to predict the package (in LPA) based on a student's CGPA.

## Dataset

The dataset used consists of two features:
- **CGPA**: The cumulative grade point average of a student.
- **Package (in LPA)**: The salary package offered to the student (in Lakhs Per Annum).

The dataset is saved in a CSV file: `package.csv`.

## Libraries Used

- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation.
- **Seaborn**: For data visualization.
- **Matplotlib**: For plotting graphs.
- **Scikit-learn**: For machine learning algorithms and utilities.

## Project Workflow

1. **Data Visualization**:
    - A scatter plot is used to visualize the relationship between CGPA and the package offered.
    - A bar plot is also used to analyze the distribution of the package with respect to CGPA.
  
    ```python
    sns.scatterplot(x=dataset["CGPA"], y=dataset["Package (in LPA)"])
    sns.barplot(x=dataset["CGPA"], y=dataset["Package (in LPA)"])
    ```

2. **Train-Test Split**:
    - The dataset is split into training and testing sets using an 80-20 split.
  
    ```python
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    ```

3. **Model Training**:
    - A linear regression model is created and trained on the training set.

    ```python
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    ```

4. **Model Prediction**:
    - The model predicts salary packages based on CGPA input.

    ```python
    lr.predict([[2.5]])
    lr.predict([[5]])
    ```

5. **Model Evaluation**:
    - The model's performance is evaluated using the test set and the R² score.
    
    ```python
    lr.score(x_test, y_test)
    ```

6. **Visualization of Prediction**:
    - The regression line is plotted along with the original data to visualize the model's predictions.
  
    ```python
    y_predict = lr.predict(x)
    plt.plot(dataset["CGPA"], y_predict, c="green")
    ```

## Results

- **Model Coefficients**: `lr.coef_ = 4.87`
- **Intercept**: `lr.intercept_ = -7.46`
- **Model Score (R²)**: `94.87%`



