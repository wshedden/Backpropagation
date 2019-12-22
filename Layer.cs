using System;

namespace Backpropagation {
    class Layer {
        public double[] inputs;
        public double[] outputs;
        public double[,] weights;
        double[,] weightDeltas;
        public double[] gamma;
        double[] error;
        static Random random = new Random();
        public Layer(int inputNum, int outputNum) {
            inputs = new double[inputNum];
            outputs = new double[outputNum];
            weights = new double[outputNum, inputNum];
            weightDeltas = new double[outputNum, inputNum];
            gamma = new double[outputNum];
            error = new double[outputNum];
            Initialise();

        }

        public void Initialise() {
            for (int i = 0; i < outputs.Length; i++) {
                for (int k = 0; k < inputs.Length; k++) {
                    weights[i, k] = random.NextDouble() * 1d - 0.5d;
                }
            }
        }

        public double[] FeedForward(double[] inputs) {
            this.inputs = inputs;
            for (int i = 0; i < outputs.Length; i++) {
                outputs[i] = 0;
                for (int k = 0; k < inputs.Length; k++) {
                    outputs[i] += inputs[k] * weights[i, k];
                }
                outputs[i] = Math.Tanh(outputs[i]);
            }
            return outputs;
        }

        public void BackpropHidden(double[] gammaForward, double[,] weightsForward) {
            for (int i = 0; i < outputs.Length; i++) {
                gamma[i] = 0;
                for (int k = 0; k < gammaForward.Length; k++) {
                    gamma[i] += gammaForward[k] * weightsForward[k, i];
                }
                gamma[i] *= Tanhderiv(outputs[i]);
            }

            for (int i = 0; i < outputs.Length; i++) {
                for (int k = 0; k < inputs.Length; k++) {
                    weightDeltas[i, k] = gamma[i] * inputs[k];
                }
            }
        }

        public void BackpropOutput(double[] expected) {
            for (int i = 0; i < outputs.Length; i++) {
                error[i] = outputs[i] - expected[i];
            }
            for (int i = 0; i < outputs.Length; i++) {
                gamma[i] = error[i] * Tanhderiv(outputs[i]);
            }
            for (int i = 0; i < outputs.Length; i++) {
                for (int k = 0; k < inputs.Length; k++) {
                    weightDeltas[i, k] = gamma[i] * inputs[k];
                }
            }
        }

        private double Tanhderiv(double value) {
            return 1d - (value * value);
        }

        public void UpdateWeights(double learningRate) {
            for (int i = 0; i < outputs.Length; i++) {
                for (int k = 0; k < inputs.Length; k++) {
                    weights[i, k] -= weightDeltas[i, k] * learningRate;
                }
            }
        }
    }
}