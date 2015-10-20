#include <stdio.h>
#include <stdlib.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <armadillo>
#include <string>
#include <fstream>
#include <sstream>

#include "ann.h"

using namespace arma;

/**
 * Store each network layer values(weights and bias)
 * each matrix is delimited by ~
 */
void storeNetwork(ANN *network, char *path) {

  ofstream file;
  file.open(path);

  // save all layers within a single file
  for(int i = 0; i < network->layerCount; i++) {
    network->layers[i]->weights.save(file, arma_ascii);
    file << "~\n";
    network->layers[i]->bias.save(file, arma_ascii);
    if(i != network->layerCount - 1) {
      file << "~\n";
    }
  }

  file.close();
}

/**
 * Load network
 */
void loadNetwork(ANN *network, char *path) {
  
  ifstream file(path);
  if(file.is_open()) {
    int k = 0;
    int layer = 0;
    mat tmpMatrix;
    string tmpMatrixString;
    stringstream tmpMatrixStream;

    while(getline(file, tmpMatrixString, '~')) {

      tmpMatrixStream.str(string());

      tmpMatrixStream << tmpMatrixString;
      tmpMatrix.load(tmpMatrixStream, arma_ascii);

      if(k == 0) {
        network->layers[layer]->weights = tmpMatrix;
      }
      else {
        network->layers[layer]->bias = tmpMatrix;
      }

      k++;
      if(k % 2 == 0) {
        k = 0;
        layer++;
      }

    }

  }

}
