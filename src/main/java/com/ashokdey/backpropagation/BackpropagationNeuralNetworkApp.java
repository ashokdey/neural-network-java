package com.ashokdey.backpropagation;

public class BackpropagationNeuralNetworkApp {
    public static void run() {
        float [][] trainingData = new float[][] {
                new float[] {0, 0},
                new float[] {0, 1},
                new float[] {1, 0},
                new float[] {1, 1},
        };

        float[][] trainingResult = new float[][] {
                new float[] {0},
                new float[] {0},
                new float[] {0},
                new float[] {1},
        };

        // create the neural network
        NeuralNetwork backpropagationNeuralNetwork = new NeuralNetwork(2, 3, 1);

        for (int iterations = 1; iterations <= Constants.ITERATIONS; ++iterations) {
            for (int i = 0; i < trainingData.length; ++i) {
                backpropagationNeuralNetwork
                        .train(trainingData[i], trainingResult[i],
                                Constants.LEARNING_RATE, Constants.MOMENTUM);
            }

            for (int i = 0; i < trainingResult.length; ++i) {
                float [] t = trainingData[i];
                System.out.println("Number of iterations = " + iterations);
                float[] result = backpropagationNeuralNetwork.run(t);
                System.out.printf("%.1f %.1f -->  %.2f (%.6f)\n",
                        t[0], t[1], result[0], result[0]);
            }
        }
    }
}
