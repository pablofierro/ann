#include <stdio.h>
#include <iostream>
#include <cmath>
#include <time.h>
#include <map>
#include <armadillo>

#include "ann/ann.h"

#define LEARNING_RATE 1.0
#define MOMENTUM_CONST 0.05

using namespace std;
using namespace arma;

// activation function
double activation_fn(double z) {
  return 1.0 / (1.0 + exp(-z));
}

// activation function first order derivative 
double activation_dev_fn(double z) {
  return (activation_fn(z) * (1 - activation_fn(z)));
}

// gets digit from network output
int getDigit(Row<double>& output) {
  uword col;
  output.max(col);
  return (int)col;
}

int main(int argc, char** argv) {

  // use random seed in generator
  arma_rng::set_seed_random();

  // activation functions
  Function fn = &activation_fn;
  Function fnDev = &activation_dev_fn;

  // create neural network, first element is input layer and last element is output layer
  int layersSize[] = {784, 30, 10};
  ANN *network = initNetwork(3, layersSize);
  setNetworkParameters(network, LEARNING_RATE, MOMENTUM_CONST, &fn, &fnDev);

  // load MNIST data
  Mat<double> inputData;
  Mat<double> outputData;
  inputData.load("./data/mnist_train_input.mat");
  outputData.load("./data/mnist_train_labels.mat");

  Mat<double> testData;
  Mat<double> testOutput;
  testData.load("./data/mnist_test_input.mat");
  testOutput.load("./data/mnist_test_labels.mat");

  trainNetwork(network, inputData, outputData, 30, 0.005, 30, testData, testOutput);

  return 0;

}
