# Real Estate Price Predictor: 5-Algorithm ML Pipeline

A full-stack machine learning pipeline built in Java that evaluates and compares five distinct regression algorithms to predict housing prices. The system processes a dataset of over 20,000 California real estate records and serves the results dynamically via a custom-built local web server.

##  System Architecture

Unlike standard terminal-based ML scripts, this project is decoupled into a robust backend processing engine and a responsive frontend dashboard.

* **Backend Engine:** Pure Java utilizing the Weka Machine Learning library for data preprocessing, model training, and cross-validation.
* **Server:** Java's native `HttpServer` handles web requests, eliminating the need for heavy external web frameworks.
* **Frontend UI:** HTML5, CSS3, and Chart.js injected directly from the Java server to visualize prediction errors in a modern, interactive bar chart.

##  Algorithms Evaluated

The pipeline benchmarks the Mean Absolute Error (MAE) of the following algorithms:
1. **Ridge Regression:** Linear regularization to establish a baseline.
2. **K-Nearest Neighbors (KNN):** Instance-based learning (K=5) to mimic real-world "comps" pricing.
3. **M5P Model Tree:** A hybrid decision tree applying linear regression at terminal nodes.
4. **Random Forest:** Parallel ensemble method utilizing bagging across 100 independent decision trees.
5. **Gradient Boosting:** Sequential ensemble method utilizing 50 decision stumps to iteratively reduce residual errors.

##  Key Results & Engineering Takeaways

* Successfully bypassed Weka's strict parsing limitations by pre-cleaning the Kaggle dataset, handling missing values and dropping non-numerical nominal columns.
* Implemented 3-fold cross-validation to ensure model reliability without overfitting.
* **Observation:** Tree-based ensemble methods (Random Forest and Gradient Boosting) significantly outperformed linear baselines, demonstrating their superior ability to map non-linear relationships in complex real estate data (such as the compounded effect of age, square footage, and bedroom count).

##  How to Run Locally

1. Clone this repository.
2. Ensure you have the `weka.jar` file in your classpath.
3. Compile and run `App.java`.
4. The terminal will log the training progress. Once complete, it will spin up the local server.
5. Open a web browser and navigate to `http://localhost:8080` to view the interactive dashboard.