# ANN

Basic multilayer artificial neural network that obtains ~94% accuracy on the MNIST test data set.
It still has many areas of improvement(momentum, matrix operations optimizations).

It is possible to run the matrix multiplication operations on the GPU by running ```make run_cuda```, this is
provided by the NVBlas library and Armadillo.

### TODO

- Mean squared error stop criteria
- Matrix operation optimizations
- Code optimizations
