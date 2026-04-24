package com.mlproject;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.RandomForest; 
import weka.classifiers.meta.AdditiveRegression; 
import weka.classifiers.trees.DecisionStump; 
import weka.classifiers.lazy.IBk; // K-Nearest Neighbors
import weka.classifiers.trees.M5P; // Model Trees
import java.io.File;
import java.util.Random;

public class App {
    public static void main(String[] args) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("housing.csv"));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); 

            System.out.println("Data Loaded! Total Houses: " + data.numInstances());
            System.out.println("Beginning 5-Algorithm Master Comparison...\n");
            
            int folds = 3; 
            Random seed = new Random(42);

            // 1. Ridge Regression (Linear)
            System.out.println("[1/5] Training Ridge Regression...");
            LinearRegression ridge = new LinearRegression();
            ridge.setRidge(2.5); 
            Evaluation evalRidge = new Evaluation(data);
            evalRidge.crossValidateModel(ridge, data, folds, seed);

            // 2. K-Nearest Neighbors (Instance-Based)
            System.out.println("[2/5] Training K-Nearest Neighbors (K=5)...");
            IBk knn = new IBk(5); // Looks at the 5 most similar houses
            Evaluation evalKNN = new Evaluation(data);
            evalKNN.crossValidateModel(knn, data, folds, seed);

            // 3. M5P Model Tree (Tree/Linear Hybrid)
            System.out.println("[3/5] Training M5P Model Tree...");
            M5P m5p = new M5P();
            Evaluation evalM5P = new Evaluation(data);
            evalM5P.crossValidateModel(m5p, data, folds, seed);

            // 4. Random Forest (Parallel Ensemble)
            System.out.println("[4/5] Training Random Forest (100 trees)...");
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100); 
            Evaluation evalRF = new Evaluation(data);
            evalRF.crossValidateModel(rf, data, folds, seed);

            // 5. Gradient Boosting (Sequential Ensemble)
            System.out.println("[5/5] Training Gradient Boosting (50 trees)...");
            AdditiveRegression gbr = new AdditiveRegression();
            gbr.setClassifier(new DecisionStump()); 
            gbr.setNumIterations(50); 
            Evaluation evalGBR = new Evaluation(data);
            evalGBR.crossValidateModel(gbr, data, folds, seed);

            // Print the Final Leaderboard
            System.out.println("\n--- Final Project Leaderboard (Mean Absolute Error) ---");
            System.out.printf("Ridge Regression  : $%.2f\n", evalRidge.meanAbsoluteError());
            System.out.printf("K-Nearest Neighbors: $%.2f\n", evalKNN.meanAbsoluteError());
            System.out.printf("M5P Model Tree    : $%.2f\n", evalM5P.meanAbsoluteError());
            System.out.printf("Random Forest     : $%.2f\n", evalRF.meanAbsoluteError());
            System.out.printf("Gradient Boosting : $%.2f\n", evalGBR.meanAbsoluteError());

        } catch (Exception e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }
}