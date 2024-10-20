package com.ashokdey.perceptron;

import com.ashokdey.ActivationFunction;

public class PerceptronNetwork {
    private final float[] weights;
    // number of edge weights
    private final int numberOfWeights;
    private final float[][] inputFeatures;
    private final float[] inputLabels;

    public PerceptronNetwork(float[][] inputFeatures, float[] inputLabels) {
        this.inputFeatures = inputFeatures;
        this.inputLabels = inputLabels;

        this.numberOfWeights = inputFeatures[0].length;
        this.weights = new float[numberOfWeights];

        this.initializeWeights();
    }

    public void train(float learningRate) {
        float totalError = 1;

        while(totalError != 0) {
            totalError = 0;
            for (int i = 0; i < inputLabels.length; ++i) {
                float calculatedOutput = calculateOutput(this.inputFeatures[i]);
                float error = inputLabels[i] - calculatedOutput;
                totalError += error;


                // update the weights based on error and learning rate
                for (int j = 0; j < this.numberOfWeights; ++j) {
                    weights[j] = weights[j] + learningRate * error * inputFeatures[i][j];
                }
            }
            System.out.println("Keep on training the neural network, error is = " + totalError);
        }
    }


    private  void initializeWeights() {
        // initialize the weights randomly from [-0.5, +0.5]
        for (int i = 0; i < this.numberOfWeights; i++) {
            this.weights[i] = (float)(Math.random() - 0.5);
        }
    }

    public float calculateOutput(float[] input) {
        float netOutput = 0f;
        for (int i = 0; i < input.length; i++) {
            netOutput += input[i] * this.weights[i];
        }
        return  ActivationFunction.activate(netOutput);
    }

}
