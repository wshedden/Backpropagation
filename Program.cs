using System;

namespace Backpropagation {
    class Program {
        static void Main(string[] args) {
            Network network = new Network(new int[]{2, 1});
            double[][] inputs = {new double[]{0, 0}, new double[]{0, 1}, new double[]{1, 0}, new double[]{1, 1}};
            double[][] expected = {new double[]{0}, new double[]{1}, new double[]{1}, new double[]{0}};
            network.Train(inputs, expected, epochs: 1000, learningRate: 0.5);
            network.MassTest(inputs);
        }
    }
}
