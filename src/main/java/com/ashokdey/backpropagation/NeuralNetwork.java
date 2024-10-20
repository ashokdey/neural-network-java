package com.ashokdey.backpropagation;

public class NeuralNetwork {
    private final Layer[] layers;

    public NeuralNetwork(
            int inputSize,
            int hiddenSize,
            int outputSize
    ) {
        this.layers = new Layer[2];
        layers[0] = new Layer(inputSize, hiddenSize);
        layers[1] = new Layer(hiddenSize, outputSize);
    }

    public Layer getLayers(int index) {
        return this.layers[index];
    }

    public float[] run(float[] inputs) {
        float[] activations = inputs;

        for (Layer layer : layers) {
            activations = layer.run(activations);
        }

        return activations;
    }

    public void train(
            float[] input,
            float[] targetOutput,
            float learningRate,
            float momentum
    ) {
        float[] calculatedOutput = this.run(input);
        float[] error = new float[calculatedOutput.length];

        for (int i = 0; i < error.length; ++i) {
            error[i] = targetOutput[i] - calculatedOutput[i];
        }

        for(int i = layers.length - 1; i >= 0; --i) {
            error = layers[i].train(error, learningRate, momentum);
        }
    }
}
