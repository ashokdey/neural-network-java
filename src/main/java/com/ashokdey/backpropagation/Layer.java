package com.ashokdey.backpropagation;

import com.ashokdey.ActivationFunction;

import java.util.Arrays;
import java.util.Random;

public class Layer {
    private final float[] input;
    private final float[] output;
    private final float[] weights;
    private final float[] dWeights;
    private final Random random;

    public Layer(int inputSize, int outputSize) {
        this.input = new float[inputSize + 1];
        this.output = new float[outputSize];

        this.weights = new float[(inputSize + 1) * outputSize]; // we have additional bias node
        this.dWeights = new float[this.weights.length];
        this.random = new Random();

        this.initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < this.weights.length; ++i) {
            this.weights[i] = (random.nextFloat() - 0.5f) * 4f; // range of weights will be [-2, +2]
        }
    }

    public float[] run(float[] inputArray) {
        System.arraycopy(inputArray, 0, this.input, 0, inputArray.length);

        // set the bias node to 1
        this.input[this.input.length - 1] = 1.0f;
        int offset = 0;

        for (int i = 0; i < this.output.length; ++i) {
            float sum = 0.0f;

            for (int j = 0; j < this.input.length; ++j) {
                sum += this.weights[offset + j] * this.input[j];
            }

            this.output[i] = ActivationFunction.sigmoid(sum);
            offset += this.input.length;
        }

        return Arrays.copyOf(this.output, this.output.length);
    }

    public float[] train(float[] error, float learningRate, float momentum) {
        int offset = 0;
        float[] nextError = new float[this.input.length];

        for (int i =0; i < this.output.length; ++i) {
            // output values are already sigmoid via run(), hence dSigmoid here
            float delta = error[i] * ActivationFunction.dSigmoid(this.output[i]);

            for (int j = 0; j < this.input.length; ++j) {
                int weightIndex = offset + j;
                nextError[j] = nextError[j] + this.weights[weightIndex] * delta;
                float dw = this.input[j] * delta * learningRate;
                this.weights[weightIndex] += this.dWeights[weightIndex] * momentum + dw;
                this.dWeights[weightIndex] = dw;
            }

            offset += this.input.length;
        }
        return nextError;
    }
}
