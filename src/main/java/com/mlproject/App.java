package com.mlproject;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.RandomForest; 
import weka.classifiers.meta.AdditiveRegression; 
import weka.classifiers.trees.DecisionStump; 
import weka.classifiers.lazy.IBk; 
import weka.classifiers.trees.M5P; 
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.Random;

public class App {
    
    static double ridgeError, knnError, m5pError, rfError, gbrError;

    public static void main(String[] args) {
        try {
            runMachineLearning();
            HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
            server.createContext("/", new DashboardHandler());
            server.setExecutor(null);
            server.start();
            System.out.println("SUCCESS! The Machine Learning Web Server is running.");
            System.out.println("Open your web browser and go to: http://localhost:8080");
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    private static void runMachineLearning() throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("housing.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); 
        int folds = 3; 
        Random seed = new Random(42);

        LinearRegression ridge = new LinearRegression();
        ridge.setRidge(2.5); 
        Evaluation evalRidge = new Evaluation(data);
        evalRidge.crossValidateModel(ridge, data, folds, seed);
        ridgeError = evalRidge.meanAbsoluteError();

        IBk knn = new IBk(5); 
        Evaluation evalKNN = new Evaluation(data);
        evalKNN.crossValidateModel(knn, data, folds, seed);
        knnError = evalKNN.meanAbsoluteError();

        M5P m5p = new M5P();
        Evaluation evalM5P = new Evaluation(data);
        evalM5P.crossValidateModel(m5p, data, folds, seed);
        m5pError = evalM5P.meanAbsoluteError();

        RandomForest rf = new RandomForest();
        rf.setNumIterations(100); 
        Evaluation evalRF = new Evaluation(data);
        evalRF.crossValidateModel(rf, data, folds, seed);
        rfError = evalRF.meanAbsoluteError();

        AdditiveRegression gbr = new AdditiveRegression();
        gbr.setClassifier(new DecisionStump()); 
        gbr.setNumIterations(50); 
        Evaluation evalGBR = new Evaluation(data);
        evalGBR.crossValidateModel(gbr, data, folds, seed);
        gbrError = evalGBR.meanAbsoluteError();
    }

    static class DashboardHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String htmlResponse = "<!DOCTYPE html>" +
                "<html><head><title>Real Estate Dashboard</title>" +
                "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>" +
                "<style>" +
                "  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; margin: 0; padding: 2vh; height: 100vh; box-sizing: border-box; overflow: hidden; }" +
                "  .container { max-width: 900px; height: 96vh; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); display: flex; flex-direction: column; box-sizing: border-box; }" +
                "  h1 { text-align: center; color: #2c3e50; margin-top: 0; margin-bottom: 1vh; }" +
                "  p { text-align: center; color: #666; margin-top: 0; margin-bottom: 2vh; }" +
                "  .chart-wrapper { position: relative; flex-grow: 1; width: 100%; min-height: 0; }" +
                "</style>" +
                "</head><body>" +
                "<div class='container'>" +
                "  <h1>Real Estate Predictor</h1>" +
                "  <p>Comparison of 5 Regression Algorithms by Mean Absolute Error (Lower is better)</p>" +
                "  <div class='chart-wrapper'><canvas id='resultsChart'></canvas></div>" +
                "</div>" +
                "<script>" +
                "  const ctx = document.getElementById('resultsChart');" +
                "  new Chart(ctx, {" +
                "    type: 'bar'," +
                "    data: {" +
                "      labels: ['Ridge', 'K-Nearest Neighbors', 'M5P Tree', 'Random Forest', 'Gradient Boosting']," +
                "      datasets: [{" +
                "        label: 'Prediction Error ($)'," +
                "        data: [" + ridgeError + ", " + knnError + ", " + m5pError + ", " + rfError + ", " + gbrError + "]," +
                "        backgroundColor: ['#ef4444', '#f97316', '#10b981', '#3b82f6', '#8b5cf6']," +
                "        borderWidth: 1" +
                "      }]" +
                "    }," +
                "    options: {" +
                "      responsive: true," +
                "      maintainAspectRatio: false," +
                "      scales: {" +
                "        y: { beginAtZero: true, title: { display: true, text: 'Mean Absolute Error (USD)' } }" +
                "      }" +
                "    }" +
                "  });" +
                "</script>" +
                "</body></html>";

            t.sendResponseHeaders(200, htmlResponse.length());
            OutputStream os = t.getResponseBody();
            os.write(htmlResponse.getBytes());
            os.close();
        }
    }
}