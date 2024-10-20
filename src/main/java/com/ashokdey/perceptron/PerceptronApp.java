package com.ashokdey.perceptron;

public class PerceptronApp {
    public static void run() {
        float [][] inputFeatures = {{0,0}, {0, 1}, {1, 0}, {1, 1}};
        float [] inputLabels = {0, 0, 0, 1};

        PerceptronNetwork network = new PerceptronNetwork(
                inputFeatures,
                inputLabels
        );

        network.train(0.02f);

        System.out.println("\nTesting network after training");

        System.out.println(network.calculateOutput(new float[] {0, 0}));
        System.out.println(network.calculateOutput(new float[] {0, 1}));
        System.out.println(network.calculateOutput(new float[] {1, 0}));
        System.out.println(network.calculateOutput(new float[] {1, 1}));
    }
}
