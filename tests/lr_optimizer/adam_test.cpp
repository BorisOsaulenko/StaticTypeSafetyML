#include "../../src/lr_optimizer/Adam.h"
#include "../../src/matrix/Matrix.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

constexpr double kTolerance = 1e-6;

class AdamTest : public testing::Test {
public:
  using json = nlohmann::json;

protected:
  const double default_lr = 1e-3;
  const double default_beta1 = 0.9;
  const double default_beta2 = 0.999;
  const double default_epsilon = 1e-8;
};

TEST_F(AdamTest, FactoryDefaultParameters) {
  const AdamFactory factory;
  Vector<3> params = {1.0, 2.0, 3.0};
  const Adam<Vector<3>> adam = factory.build(params);

  EXPECT_EQ(adam.get_learning_rate(), default_lr);
}

TEST_F(AdamTest, FactoryCustomParameters) {
  const double custom_lr = 0.01;
  const double custom_beta1 = 0.95;
  const double custom_beta2 = 0.99;
  const double custom_epsilon = 1e-6;

  const AdamFactory factory(custom_lr, custom_beta1, custom_beta2,
                            custom_epsilon);
  Vector<3> params = {1.0, 2.0, 3.0};
  const Adam<Vector<3>> adam = factory.build(params);

  EXPECT_EQ(adam.get_learning_rate(), custom_lr);
}

TEST_F(AdamTest, AdjustVector_SingleStep) {
  Vector<3> params = {1.0, 2.0, 3.0};
  Vector<3> gradient = {0.1, 0.2, 0.3};

  AdamFactory factory(0.001, 0.9, 0.999, 1e-8);
  Adam<Vector<3>> adam = factory.build(params);

  const Vector<3> initial_params = params;
  adam.adjust(gradient);

  EXPECT_NE(params[0], initial_params[0]);
  EXPECT_NE(params[1], initial_params[1]);
  EXPECT_NE(params[2], initial_params[2]);
  EXPECT_LT(params[0], initial_params[0]);
  EXPECT_LT(params[1], initial_params[1]);
  EXPECT_LT(params[2], initial_params[2]);
}

TEST_F(AdamTest, AdjustVector_MultipleSteps) {
  Vector<3> params = {1.0, 2.0, 3.0};
  Vector<3> gradient = {0.1, 0.2, 0.3};

  AdamFactory factory(0.001, 0.9, 0.999, 1e-8);
  Adam<Vector<3>> adam = factory.build(params);

  Vector<3> prev_params = params;

  for (int i = 0; i < 5; ++i) {
    adam.adjust(gradient);
    EXPECT_NE(params, prev_params) << "Parameters should change at step " << i;
    prev_params = params;
  }
}

TEST_F(AdamTest, AdjustMatrix_SingleStep) {
  Matrix<2, 2> params = {1.0, 2.0, 3.0, 4.0};
  Matrix<2, 2> gradient = {0.1, 0.2, 0.3, 0.4};

  AdamFactory factory(0.001, 0.9, 0.999, 1e-8);
  Adam<Matrix<2, 2>> adam = factory.build(params);

  const Matrix<2, 2> initial_params = params;
  adam.adjust(gradient);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      // do not put directly into EXPECT_NE macro. Will break
      const double actual_param = params[i, j];
      const double initial_param = initial_params[i, j];
      EXPECT_NE(actual_param, initial_param);
      EXPECT_LT(actual_param, initial_param);
    }
  }
}

TEST_F(AdamTest, ZeroGradient_NoParameterUpdate) {
  Vector<3> params = {1.0, 2.0, 3.0};
  Vector<3> gradient = {0.0, 0.0, 0.0};

  AdamFactory factory(0.001, 0.9, 0.999, 1e-8);
  Adam<Vector<3>> adam = factory.build(params);

  const Vector<3> initial_params = params;
  adam.adjust(gradient);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(params[i], initial_params[i], kTolerance);
  }
}

TEST_F(AdamTest, PositiveAndNegativeGradients) {
  Vector<3> params = {1.0, 2.0, 3.0};
  Vector<3> gradient = {-0.1, 0.2, -0.3};

  AdamFactory factory(0.001, 0.9, 0.999, 1e-8);
  Adam<Vector<3>> adam = factory.build(params);

  const Vector<3> initial_params = params;
  adam.adjust(gradient);

  EXPECT_GT(params[0], initial_params[0]);
  EXPECT_LT(params[1], initial_params[1]);
  EXPECT_GT(params[2], initial_params[2]);
}

TEST_F(AdamTest, LargerLearningRateHasLargerEffect) {
  Vector<3> params1 = {1.0, 2.0, 3.0};
  Vector<3> params2 = {1.0, 2.0, 3.0};
  Vector<3> gradient = {0.1, 0.2, 0.3};

  AdamFactory factory1(0.001, 0.9, 0.999, 1e-8);
  AdamFactory factory2(0.01, 0.9, 0.999, 1e-8);

  Adam<Vector<3>> adam1 = factory1.build(params1);
  Adam<Vector<3>> adam2 = factory2.build(params2);

  adam1.adjust(gradient);
  adam2.adjust(gradient);

  for (int i = 0; i < 3; ++i) {
    EXPECT_LT(params2[i], params1[i]);
  }
}

TEST_F(AdamTest, ConsistentConvergence) {
  Vector<2> params = {10.0, 10.0};
  Vector<2> gradient = {1.0, 1.0};

  AdamFactory factory(0.1, 0.9, 0.999, 1e-8);
  Adam<Vector<2>> adam = factory.build(params);

  for (int i = 0; i < 100; ++i) {
    adam.adjust(gradient);
  }

  EXPECT_LT(params[0], 10.0);
  EXPECT_LT(params[1], 10.0);
}

TEST_F(AdamTest, SerializationDeserialization_Vector) {
  Vector<3> params = {1.0, 2.0, 3.0};
  Vector<3> gradient = {0.1, 0.2, 0.3};

  AdamFactory factory(0.001, 0.9, 0.999, 1e-8);
  Adam<Vector<3>> adam = factory.build(params);

  adam.adjust(gradient);
  adam.adjust(gradient);

  json j;
  to_json(j, adam);

  Vector<3> params_restored = {1.0, 2.0, 3.0};
  Adam<Vector<3>> adam_restored = factory.build(params_restored);
  from_json(j, adam_restored);

  EXPECT_NEAR(adam.get_learning_rate(), adam_restored.get_learning_rate(),
              kTolerance);
}

TEST_F(AdamTest, SerializationDeserialization_Matrix) {
  Matrix<2, 2> params = {1.0, 2.0, 3.0, 4.0};
  Matrix<2, 2> gradient = {0.1, 0.2, 0.3, 0.4};

  AdamFactory factory(0.001, 0.9, 0.999, 1e-8);
  Adam<Matrix<2, 2>> adam = factory.build(params);

  adam.adjust(gradient);
  adam.adjust(gradient);

  json j;
  to_json(j, adam);

  Matrix<2, 2> params_restored = {1.0, 2.0, 3.0, 4.0};
  Adam<Matrix<2, 2>> adam_restored = factory.build(params_restored);
  from_json(j, adam_restored);

  EXPECT_NEAR(adam.get_learning_rate(), adam_restored.get_learning_rate(),
              kTolerance);
}
