#include <stdio.h>
#include <stdlib.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <armadillo>

#include "ann.h"

using namespace arma;

void feedForward(ANN *network, Mat<double> *initialInput, vector<Mat<double> *>& z, vector<Mat<double> *>& output);

/**
 * Initialize layer
 */
Layer *initLayer(int type, int nodeCount, int prevCount) {

  // allocate new layer
  Layer *layer = new Layer();
  layer->nodeCount = nodeCount;
  layer->type = type;

  // bias vector for each layer
  layer->bias.randn(nodeCount);

  // weights matrix for each layer
  layer->weights.randn(nodeCount, prevCount);

  return layer;
}

/**
 * Initializes a network
 */
ANN *initNetwork(int layerCount, int layerSizes[]) {

  // initialize network
  ANN *network = new ANN;
  Layer **layers = new Layer*[layerCount];
  network->layerCount = layerCount - 1;

  // initialize hidden layers
  for(int i = 0; i < layerCount - 1; i++) {
    layers[i] = initLayer((i == layerCount - 2 ? 2 : 1), layerSizes[i + 1], layerSizes[i]);
  }

  network->layers = layers;

  return network;
}

/**
 * Set network hyper-parameters and activation functions
 */
void setNetworkParameters(ANN *network, double learningRate, double alpha, Function *fn, Function *fnDev) {
  network->learningRate= learningRate;
  network->momentum = alpha;
  network->sigmaFn = fn;
  network->sigmaPrimeFn = fnDev;
}

/**
 * Get mean standard error
 * TODO: implement this stoping criteria
 */
double getMSE(ANN *network) {
  return 0;
}

/**
 * Run input on network
 */
void runInput(ANN *network, Row<double>& input, Row<double>& output) {

  vector<Mat<double> *> layersZ(network->layerCount);
  vector<Mat<double> *> layersOutput(network->layerCount);

  for(int i = 0; i < network->layerCount; i++) {
    layersZ[i] = new Mat<double>(1, network->layers[i]->nodeCount, fill::zeros);
    layersOutput[i] = new Mat<double>(1, network->layers[i]->nodeCount, fill::zeros);
  }

  Mat<double>& outputVector = *layersOutput[network->layerCount - 1];

  feedForward(network, &input, layersZ, layersOutput);

  output = outputVector.row(0);

}
