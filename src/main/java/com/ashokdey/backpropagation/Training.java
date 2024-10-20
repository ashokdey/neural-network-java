package com.ashokdey.backpropagation;

public class Training {
    public static float[][][] TrainingData() {
        // data will be the possible combination of 0,1
        return new float[][][] {
            new float[][] {
                new float[] {0, 0},
                new float[] {0, 1},
                new float[] {1, 0},
                new float[] {1, 1},
            },
            new float[][] {
                new float[] {0, 0},
                new float[] {0, 1},
                new float[] {1, 0},
                new float[] {1, 1},
            } ,
            new float[][] {
                new float[] {0, 0},
                new float[] {0, 1},
                new float[] {1, 0},
                new float[] {1, 1},
            }
        };
    }

    public static float[][][] TrainingResult() {
        return new float[][][] {
           // AND
           new float[][] {
                new float[] {0},
                new float[] {0},
                new float[] {0},
                new float[] {1},
           },
           // OR
           new float[][] {
                new float[] {0},
                new float[] {1},
                new float[] {1},
                new float[] {1},
           },
           // XOR
           new float[][] {
                new float[] {0},
                new float[] {1},
                new float[] {1},
                new float[] {0},
           },
        };
    }
}
