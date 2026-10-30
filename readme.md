# ML: a home-made C++ neural network framework

A small machine-learning framework written from scratch in modern C++ (C++26).
It implements its own linear algebra, layers, activations, losses, optimizers and a
training loop. There is no BLAS, no autograd library and no Python.

> **This is an educational project.** It exists to understand how neural networks
> work all the way down, and to see how far the C++ type system can go in catching
> mistakes before the program runs. It is not meant to compete with PyTorch,
> TensorFlow or Eigen, and it has known inefficiencies (see below).

## Philosophy

### Learn by building everything
Every piece is written by hand and small enough to read in one sitting:
forward and backward passes for each layer, gradient accumulation, Adam,
cross-entropy, JSON serialization of the trained network.

### Static type safety
Shapes are part of the type. A `Matrix<3, 4>` and a `Matrix<4, 3>` are different
types, and the layers carry their dimensions as template parameters:

```cpp
auto conv1   = Convolutional<1, 5, 5, 8>(ActivationFunctionEnum::Sigmoid);
auto pool1   = MaxPool2x2();
auto conv2   = Convolutional<8, 5, 5, 16>(ActivationFunctionEnum::Sigmoid);
auto pool2   = MaxPool2x2();
auto flatten = Flatten();
auto dense1  = DenseLayer<16 * 4 * 4, 128>(ActivationFunctionEnum::Sigmoid);
auto dense2  = DenseLayer<128, 10>(ActivationFunctionEnum::Identity);

AdamFactory optimizer_factory(0.01);
auto network = Network(optimizer_factory, LossEnum::CrossEntropy,
                       conv1, pool1, conv2, pool2, flatten, dense1, dense2);
```

When the network is used, `Network` works out each layer's input and output type at
compile time and checks with `static_assert` that:

- each layer accepts the previous layer's output,
- each layer's backward pass (`compute_delta`) is consistent with its forward pass (`compute_z`),
- the training targets have the network's output type,
- the training set size is divisible by the batch size.

A shape mismatch, such as `DenseLayer<256, 128>` after a flatten that produces 255
values, is a **compile error** that names the offending layer. It is not a runtime crash
halfway through an epoch. Layers, optimizers and the algebra types are described with
C++20 concepts (`Algebraic`, `ComputesZ`, `ComputesDeltaCached`, `OptimizerFactoryFor`, ...).

## What's inside

| Area | Contents |
|------|----------|
| `src/matrix/` | `Matrix<R, C>`, `Vector<N>`, `Tensor3D<Ch, R, C>`, `StaticArray<N, T>`, `AlgebraicTuple`, `Algebraic` concept |
| `src/layers/` | `DenseLayer`, `Convolutional`, `MaxPool2x2`, `Flatten`; base classes `ParamsLayer` / `NoParamsLayer` |
| `src/activation/` | Identity, Sigmoid, ReLU |
| `src/loss/` | MSE, Cross-entropy (softmax) |
| `src/lr_optimizer/` | Adam, Gradient descent |
| `src/Network.h` | Compile-time layer composition, batched training, prediction, JSON save/load |
| `include/mnist/` | MNIST readers (vector and tensor form) |
| `tests/` | GoogleTest unit tests |

Work in progress (towards a small language model): `Embedding`, `AttentionHead`,
and a BPE tokenizer in `src/preprocessing/tokenizer.h`.

## Building

Requirements:
- CMake 3.20+
- A C++26-capable compiler (uses pack indexing, `constexpr std::string`, etc.)
- OpenMP
- nlohmann/json 3.12+
- STB (only for image display on mnist example)
- Network access on the first configure (GoogleTest is fetched)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ML -j          # MNIST CNN demo (main.cpp)
cmake --build build --target hello_test -j  # unit tests
ctest --test-dir build --output-on-failure
```

The build treats warnings as errors (`-Wall -Wextra -Werror`).

## Example: MNIST

`main.cpp` trains a small CNN (conv → pool → conv → pool → dense → dense) on the
bundled MNIST data in `datasets/mnist/`, prints the training and validation loss per
epoch, saves the weights to `network_mnist_cnn.json`, and reports train and test accuracy.
`examples/mnist.cpp` shows a plain two-layer dense network.

## Known inefficiencies

These are known and accepted for now. Readability and type safety come first.

- **Naive matrix multiplication.** `operator*` is a plain triple loop, roughly 2 GFLOP/s
  in FP64 on an M1 core. A blocked/packed `std::simd` kernel is planned.
- **Heap storage behind fixed-size types.** Sizes are compile-time, but `Matrix` and
  `StaticArray` keep their data in a `std::vector`, so every temporary allocates.
- **Lots of copies.** Getters such as `get_weights()` return by value, and the
  forward/backward passes create many intermediate matrices.
- **Compile time grows with the dataset.** The training loop unrolls over batches at
  compile time (`constexpr_for`), so large `TRAIN_SZ / BATCH_SZ` ratios slow the
  compiler down a lot.
- **CPU only, FP64 only.** Parallelism is plain OpenMP over the batch inside the layers.
  There is no GPU support.
