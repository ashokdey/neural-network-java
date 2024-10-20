package com.ashokdey;

public class ActivationFunction {
    public static int activate(float x) {
        if (x < 1) return 0;
        return 1;
    }

    public static float sigmoid(float x) {
        return (float) (1 /(1 + Math.exp(-x)));
    }

    public static float dSigmoid(float x) {
        return x * (1 - x);
    }
}
