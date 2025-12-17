# Spaceship Titanic – Machine Learning Project

## Overview

This project tackles the **Kaggle Spaceship Titanic** problem: predicting whether a passenger was **transported to another dimension** based on demographic, travel, and spending data. It’s a classic **binary classification** task with messy real‑world data (missing values, mixed feature types), making it a solid testbed for end‑to‑end ML fundamentals.

The notebook walks through **EDA → preprocessing → model training → evaluation**, with a focus on understanding feature distributions and building a baseline model that actually works.

---

## Dataset

* **Source:** Kaggle – *Spaceship Titanic*
* **Target:** `Transported` (boolean)
* **Key Features:**

  * Categorical: `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`
  * Numerical: `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`

The data contains **significant missing values** and **skewed numerical distributions**, which are explicitly explored and handled in the notebook.

---

## Exploratory Data Analysis (EDA)

The notebook performs:

* Value counts for categorical variables
* Missing value analysis
* Data type inspection
* KDE plots for numerical features (`Age`, spending variables)

This step is used to:

* Understand feature distributions
* Identify skewness and sparsity
* Inform preprocessing choices

---

## Preprocessing

Key preprocessing steps include:

* Handling missing values
* Encoding categorical variables using `sklearn.preprocessing`
* Feature scaling where appropriate
* Train–test split

All transformations are applied **before model training** to avoid data leakage.

---

## Model

* **Algorithm:** Logistic Regression
* **Configuration:**

  * `penalty='l1'`
  * `solver='saga'`
  * `class_weight='balanced'`
  * `max_iter=5000`

Why Logistic Regression?

* Strong, interpretable baseline
* Handles high‑dimensional encoded features well
* L1 regularization encourages sparsity and feature selection

---

## Evaluation

* **Metric:** Accuracy
* Predictions are generated on the test set and evaluated using `accuracy_score`.

This provides a clean baseline that can be improved with more complex models.

---

## Project Structure

```
Spaceship_Titanic.ipynb   # Main notebook (EDA, preprocessing, modeling)
README.md                # Project documentation
```

---

## How to Run

1. Download the dataset from Kaggle.
2. Update the CSV path in the notebook:

   ```python
   df = pd.read_csv('Spaceship Titanic train.csv')
   ```
3. Run the notebook top‑to‑bottom.

---

## Improvements & Next Steps

* Try tree‑based models (Random Forest, XGBoost, LightGBM)
* Perform feature engineering on `Cabin` (deck / side extraction)
* Use cross‑validation instead of a single train–test split
* Optimize hyperparameters
* Track experiments properly (notebooks don’t scale)

---

## Takeaway

This project demonstrates a **clean baseline ML workflow**. It’s not fancy, but it’s correct—and that’s the point. Once the fundamentals are solid, complexity can be added without breaking everything.
