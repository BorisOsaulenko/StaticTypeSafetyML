# AGENTS.md

## Project overview

This is a small C++ machine-learning project built around fixed-size, template-based algebra types. The main components are:

- `src/matrix/`: `Matrix`, `Vector`, `Tensor3D`, `AlgebraicArray`, and related concepts/spans.
- `src/layers/`: dense, convolutional, flatten, and max-pooling layers.
- `src/activation/`: identity, sigmoid, and ReLU activations.
- `src/loss/`: MSE and cross-entropy losses.
- `src/lr_optimizer/`: optimizer abstractions, Adam, and gradient descent.
- `src/Network.h`: compile-time layer composition, forward propagation, training, and serialization.
- `tests/`: GoogleTest unit tests for layers, matrices, and optimizers.
- `include/mnist/` and `datasets/mnist/`: MNIST readers and local dataset files.

The project is currently under active refactoring. Some APIs and tests may be transitional; inspect the current headers and nearby tests before extending them.

## Build and test

Requirements include CMake 3.20+, a C++26-capable compiler, OpenMP, OpenCV, and network access on the first configure so CMake can fetch GoogleTest and nlohmann/json.

The configured build directory is `build/`. Typical commands are:

```sh
cmake --build build --target hello_test -j2
ctest --test-dir build --output-on-failure
```

To run one test binary or suite:

```sh
./build/hello_test --gtest_filter='DenseLayerTest.*'
```

The main application target is `ML`:

```sh
cmake --build build --target ML -j2
```

The build uses C++26 and treats warnings as errors (`-Wall -Wextra -Werror`).

## Coding and testing conventions

- Prefer the existing fixed-size algebra types and layer APIs over introducing dynamic containers.
- Layer dimensions are template parameters, and parameters are generally supplied as weights/biases plus an `ActivationFunctionEnum` and `AdamFactory`.
- Use the helper functions in `tests/layers/utils.h` for approximate matrix, tensor, and vector comparisons. Explicit template dimensions may be needed because the helpers use `std::size_t` while layer dimensions use `int`.
- Keep deterministic test parameters and inputs explicit; use tolerances appropriate to the numerical operation.
- Add new tests to the executable in `CMakeLists.txt` when the source is not already listed.
- Prefer focused test filters during development, then run the full `ctest` suite when practical.

## Repository hygiene

- Preserve unrelated user changes in the worktree.
- Do not edit generated files under `build/`; rebuilds may update them automatically.
- Keep changes scoped to the requested source or test files. Avoid changing datasets or generated network JSON unless explicitly requested.
- Headers are included with project-relative paths, and most implementation is header-only.

## Useful implementation details

- `DenseLayer<i, o>` stores an `o x i` weight matrix and an `o`-element bias vector.
- `ParamsLayer` owns parameter gradients and provides vectorized forward/backward operations.
- `AlgebraicSpan` and `AlgebraicArray` are used for compile-time-sized batches.
- OpenMP is used inside vectorized layer operations; parallel test behavior should not depend on iteration order.
- `main.cpp` demonstrates a small MNIST training setup and expects the bundled dataset paths to be available.
