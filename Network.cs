using System;

namespace Backpropagation {
    class Network {
        int[] dimensions;
        Layer[] layers;
        public Network(int[] dimensions){
            this.dimensions = dimensions;
            layers = new Layer[dimensions.Length-1];
            for(int i = 0; i < layers.Length; i++){
                layers[i] = new Layer(dimensions[i], dimensions[i+1]);
            }
        }

        public double[] FeedForward (double[] inputs) {
            layers[0].FeedForward(inputs);
            for(int i = 1; i < layers.Length; i++){
                layers[i].FeedForward(layers[i-1].outputs);
            }
            return layers[layers.Length-1].outputs;

        }

        public void BackProp(double[] expected, double learningRate){
            for(int i = layers.Length-1; i >= 0; i--){
                if(i == layers.Length-1){
                    layers[i].BackpropOutput(expected);
                } else {
                    layers[i].BackpropHidden(layers[i+1].gamma, layers[i+1].weights);
                }
            }
            for(int i = 0; i < layers.Length; i++){
                layers[i].UpdateWeights(learningRate: learningRate);
            }
        }

        public void Train(double[][] inputs, double[][] expected, int epochs = 500, double learningRate = 0.05d) {
            for(int i = 0; i < epochs; i++){
                for(int k = 0; k < inputs.Length; k++){
                    FeedForward(inputs[k]);
                    BackProp(expected[k], learningRate);
                }
            }
        }

        public void MassTest(double[][] inputs){
            for(int i = 0; i < inputs.Length; i++){
                layers[0].inputs = inputs[i];
                double[] output = FeedForward(inputs[i]);
                for(int k = 0; k < output.Length; k++){
                    Console.WriteLine($"Output[{i}][{k}] = {output[k]}");
                }
            }
        }
    }
}