#include <stdio.h>
#include <stdlib.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <armadillo>

#include "ann.h"

using namespace arma;

// private extern functions
void mapTo(Mat<double>&, Mat<double>&, Function *);
double getMSE(ANN *);

/**
 * Maps a matrix applying a transform function
 */
void mapTo(Mat<double>& target, Mat<double>& source, Function *fn) {
  int n = source.n_elem;
  while(n--) {
    target[n] = (*fn)(source[n]);
  }
}

/**
 * Propagate input through the network
 */
void feedForward(ANN *network, Mat<double> *initialInput, vector<Mat<double> *>& z, vector<Mat<double> *>& output) {
  Mat<double> *input = initialInput;
  for(int l = 0; l < network->layerCount; l++) {
    *z[l] = *input * network->layers[l]->weights.t() + (repmat(network->layers[l]->bias, (*z[l]).n_rows, 1));
    mapTo(*output[l], *z[l], network->sigmaFn);
    input =  output[l];
  }
}

/**
 * Calculate nablas
 */
void feedBackwards(ANN *network, Mat<double> *expected, vector<Mat<double> *>& delta, vector<Mat<double> *>& output, vector<Mat<double> *>& outputPrime, vector<Mat<double> *>& z) {

  Layer *tmpLayer;
  Layer *tmpNextLayer;
  for(int l = network->layerCount - 1; l >= 0; l--) {
    tmpLayer = network->layers[l];
    mapTo(*outputPrime[l], *z[l], network->sigmaPrimeFn);

    // output layer delta
    if(tmpLayer->type == 2) {
      *delta[l] = (*output[l] - *expected) % *outputPrime[l];
    }

    // hidden layer delta
    else {
      tmpNextLayer = network->layers[l + 1];
      *delta[l] = (*delta[l+1]) * tmpNextLayer->weights % *outputPrime[l];
    }

  }

}

/**
 * Change weights
 */
void changeWeights(ANN *network, Mat<double> *initialInput, vector<Mat<double> *>& output, vector<Mat<double> *>& delta) {

  Mat<double> *input = initialInput;
  Layer *tmpLayer;
  for(int l = 0; l < network->layerCount; l++) {

    tmpLayer = network->layers[l];

    tmpLayer->weights += (*delta[l]).t() * (*input) * (-network->learningRate / (*input).n_rows);

    network->layers[l]->bias += (-network->learningRate / (*input).n_rows) * accu(*delta[l]);

    input = output[l];

  }

}

/**
 * Shuffles a training data set, input and output must
 * have the same number of rows
 */
void shuffleEntries(mat& inputs, mat& output) {
  int tmpIndex;
  for(int i = 0; i < inputs.n_rows; i++) {
    tmpIndex = rand() % inputs.n_rows;
    inputs.swap_rows(i, tmpIndex);
    output.swap_rows(i, tmpIndex);
  }
}

/**
 * Train network
 */
void trainNetwork(ANN *network, mat& inputs, mat& outputs, int maxEpochs, double accuracy, int batchSize, mat& testInputs, mat& testOutputs) {

  double error = std::numeric_limits<double>::infinity();
  double errorSum = 0;
  int epochs = 0;

  // initialize training data
  vector<Mat<double> *> layersZ(network->layerCount);
  vector<Mat<double> *> layersOutput(network->layerCount);
  vector<Mat<double> *> layersDelta(network->layerCount);
  vector<Mat<double> *> layersOutputPrime(network->layerCount);

  for(int i = 0; i < network->layerCount; i++) {
    layersZ[i] = new Mat<double>(batchSize, network->layers[i]->nodeCount, fill::zeros);
    layersOutput[i] = new Mat<double>(batchSize, network->layers[i]->nodeCount, fill::zeros);
    layersOutputPrime[i] = new Mat<double>(batchSize, network->layers[i]->nodeCount, fill::zeros);
    layersDelta[i] = new Mat<double>(batchSize, network->layers[i]->nodeCount, fill::zeros);
  }

  Mat<double> inputBatch(batchSize, inputs.n_cols);
  Mat<double> outputBatch(batchSize, outputs.n_cols);

  while(epochs < maxEpochs && error > accuracy) {

    // shuffle data
    shuffleEntries(inputs, outputs);

    // iterate all batches
    for(int i = 0; i < inputs.n_rows; i += batchSize) {

      // assign input
      inputBatch = inputs.rows(i, i + batchSize - 1);
      outputBatch = outputs.rows(i, i + batchSize - 1);

      feedForward(network, &inputBatch, layersZ, layersOutput);
      feedBackwards(network, &outputBatch, layersDelta, layersOutput, layersOutputPrime, layersZ);
      changeWeights(network, &inputBatch, layersOutput, layersDelta);

    }

    validateNetwork(network, testInputs, testOutputs, 0.5);

    epochs++;

    printf("Epoch %d\n", epochs);

  }

}

/**
 * Validate network with test data
 */
void validateNetwork(ANN *network, mat& inputs, mat& output, double threshold) {

  // iterate all entries
  int n = inputs.n_rows;
  int successCount = 0;
  Mat<double> currentInput(1, inputs.n_cols);
  Mat<double> currentExpected(1, output.n_cols);

  // initialize validation network data
  vector<Mat<double> *> layersZ(network->layerCount);
  vector<Mat<double> *> layersOutput(network->layerCount);

  for(int i = 0; i < network->layerCount; i++) {
    layersZ[i] = new Mat<double>(1, network->layers[i]->nodeCount, fill::zeros);
    layersOutput[i] = new Mat<double>(1, network->layers[i]->nodeCount, fill::zeros);
  }

  Mat<double>& outputVector = *layersOutput[network->layerCount - 1];

  // iterate all data points
  for(int i = 0; i < n; i++) {

    currentInput = inputs.row(i);
    currentExpected = output.row(i);

    feedForward(network, &currentInput, layersZ, layersOutput);

    // check expected output
    bool success = true;
    for(int j = output.n_cols - 1; j >= 0; j--) {
      if((outputVector(0, j) > threshold ? 1 : 0) != currentExpected(0, j)) {
        success = false;
        break;
      }
    }

    if(success) successCount++;

  }

  if(successCount != 0) {
    printf("Sucess Rate: %.2f%%\n", ((double)successCount / (double)n) * 100);
  }

}
