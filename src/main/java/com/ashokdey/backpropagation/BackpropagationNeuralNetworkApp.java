package com.ashokdey.backpropagation;

import java.util.HashMap;
import java.util.Map;

public class BackpropagationNeuralNetworkApp {
    public static void run() {
        float[][][] trainingData = Training.TrainingData();
        float[][][] trainingResult = Training.TrainingResult();

        Map<Integer, String> networkNameMap = new HashMap<Integer, String>();
        networkNameMap.put(0, "AND Gate");
        networkNameMap.put(1, "OR Gate");
        networkNameMap.put(2, "XOR Gate");

        for (int t = 0; t < 3; ++t) {
            String networkName = networkNameMap.get(t);

            // create the neural network for AND, OR and XOR
            NeuralNetwork backpropagationNeuralNetwork = new NeuralNetwork(2, 3, 1);

            for (int iterations = 1; iterations <= Constants.ITERATIONS; ++iterations) {
                System.out.printf("Number of iterations = %d [%s]\n", iterations, networkName);

                for (int i = 0; i < trainingData[t].length; ++i) {
                    backpropagationNeuralNetwork
                            .train(trainingData[t][i], trainingResult[t][i],
                                    Constants.LEARNING_RATE, Constants.MOMENTUM);
                }

                for (int i = 0; i < trainingResult[t].length; ++i) {
                    float [] row = trainingData[t][i];
                    float[] result = backpropagationNeuralNetwork.run(row);
                    System.out.printf("%.1f %.1f -->  %.2f (%.6f)\n",
                            row[0], row[1], result[0], result[0]);
                }
                System.out.println();
            }
        }
    }
}
