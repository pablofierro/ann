#include <map>
#include <armadillo>

using namespace std;
using namespace arma;

/**
 * External activation function
 */
typedef double (*Function)(double);

/**
 * Network layer
 */
struct Layer {
  Mat<double> weights;
  Row<double> bias;
  Mat<double> v;
  int type;
  int weightCount;
  int nodeCount;
  int number;
};

/**
 * Network
 */
typedef struct {
  Mat<double> errors;
  double momentum;
  double learningRate;
  int outputNodes;
  int layerCount;
  int batchSize;
  Layer **layers;
  Function *sigmaFn;
  Function *sigmaPrimeFn;
} ANN;

/**
 * Network methods
 */
ANN *initNetwork(int layerCount, int layers[]);
void setNetworkParameters(ANN *network, double learningRate, double alpha, Function *fn, Function *fnDev);
void trainNetwork(ANN *network, mat& inputs, mat& output, int maxEpochs, double accuracy, int batchSize, mat& testInputs, mat& testOutput);
void runInput(ANN *network, Row<double>& input, Row<double>& output);
void storeNetwork(ANN *network, char *path);
void loadNetwork(ANN *network, char *path);
void validateNetwork(ANN *network, mat& inputs, mat& output, double threshold);
