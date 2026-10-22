#include "../../src/Network.h"
#include "../../src/layers/Dense.h"
#include <gtest/gtest.h>

class NetworkTestFixture : public testing::Test {
public:
  // Layer 1: each hidden unit adds one pair of inputs, h_r = x[2r] + x[2r+1]
  const DenseLayer<10, 5>::Params params1{
      Matrix<5, 10>{
          1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
      },
      Vector<5>{0, 0, 0, 0, 0},
  };

  // Layer 2: output 0 = sum(h) - 5, output 1 = alternating sum(h) + 1
  const DenseLayer<5, 2>::Params params2{
      Matrix<2, 5>{
          1,
          1,
          1,
          1,
          1,
          1,
          -1,
          1,
          -1,
          1,
      },
      Vector<2>{-5, 1},
  };

  AdamFactory optimizer_factory;
  DenseLayer<10, 5> dense1 = DenseLayer<10, 5>(
      params1, ActivationFunctionEnum::Identity, optimizer_factory);
  DenseLayer<5, 2> dense2 = DenseLayer<5, 2>(
      params2, ActivationFunctionEnum::Identity, optimizer_factory);
  Network<DenseLayer<10, 5>, DenseLayer<5, 2>> network =
      Network(LossEnum::MSE, dense1, dense2);

  Vector<10> input_elem = {0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5};
  StaticArray<1, Vector<10>> input_batch = {input_elem};

  Vector<2> output_elem = {10, 4};
  StaticArray<1, Vector<2>> output_batch = {output_elem};
};

TEST_F(NetworkTestFixture, LossTypeDefined) {
  LossEnum lossType = network.get_loss_type();
  EXPECT_EQ(lossType, LossEnum::MSE);
}

TEST_F(NetworkTestFixture, NetworkJson) {
  AdamFactory other_factory;

  DenseLayer<10, 5> dense1_other(ActivationFunctionEnum::Sigmoid,
                                 other_factory);
  DenseLayer<5, 2> dense2_other(ActivationFunctionEnum::ReLu, other_factory);
  Network<DenseLayer<10, 5>, DenseLayer<5, 2>> other_network(
      LossEnum::CrossEntropy, dense1_other, dense2_other);
  std::string filename = "network_json_test.json";
  network.save(filename);
  other_network.load(filename);

  EXPECT_EQ(other_network.get_loss_type(), network.get_loss_type());
  // Smoke test as of now: compiles and gets the correct loss type
}

TEST_F(NetworkTestFixture, PredictsSingleInput) {
  const Vector<2> actual = network.predict(input_elem);
  EXPECT_EQ(actual, output_elem);
}

TEST_F(NetworkTestFixture, PredictBatchInput) {
  const StaticArray<1, Vector<2>> actual_batched =
      network.predict_vectorized(input_batch);
  EXPECT_EQ(actual_batched[0], output_batch[0]);
  EXPECT_EQ(actual_batched.size(), 1);
}