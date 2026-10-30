cmake_minimum_required(VERSION 3.20)
project(ML LANGUAGES CXX)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

# ----------------------------
# C++ configuration
# ----------------------------
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv5")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_compile_options(-Wall -Wextra -Werror -fno-elide-type -fno-omit-frame-pointer)

# ----------------------------
# GoogleTest (FetchContent)
# ----------------------------
include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

# ----------------------------
# stb (FetchContent)
# ----------------------------

FetchContent_Declare(stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master)   # better: pin a commit hash
FetchContent_MakeAvailable(stb)

# ----------------------------
# Packages 
# ----------------------------

find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

# ----------------------------
# Test target
# ----------------------------
add_executable(hello_test        
        tests/layers/dense_test.cpp
        tests/layers/convolutional_layer_test.cpp
        tests/layers/max_pool_2x2_test.cpp
        tests/layers/flatten_test.cpp
        tests/layers/matrix_test.cpp

        tests/network/network_test.cpp
        tests/lr_optimizer/adam_test.cpp
        tests/linear_regression_test.cpp

        src/layers/Embedding.hpp
        src/layers/ParamsLayer.hpp
        src/layers/Flatten.hpp
        src/layers/AttentionHead.hpp

        src/loss/Loss.hpp
        src/loss/MSE.hpp
        src/loss/LossFactory.hpp

        src/lr_optimizer/Adam.hpp
        src/lr_optimizer/Optimizer.hpp

        src/matrix/Algebraic.hpp
        src/matrix/StaticArray.hpp
        src/preprocessing/tokenizer.hpp

        tests/utils.hpp
)

target_link_libraries(hello_test
        PRIVATE
        GTest::gtest_main
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
)

include(GoogleTest)
gtest_discover_tests(hello_test)

# ----------------------------
# Main application target
# ----------------------------
add_executable(ML

        main.cpp

        src/layers/NoParamsLayer.hpp
        src/layers/Dense.hpp
        src/layers/Convolutional.hpp
        src/layers/Flatten.hpp
        src/layers/MaxPool2x2.hpp

        src/activation/ActivationFunction.hpp

        src/activation/Sigmoid.hpp
        src/activation/ReLu.hpp
        src/activation/Identity.hpp

        src/lr_optimizer/Optimizer.hpp
        src/lr_optimizer/Adam.hpp
        src/lr_optimizer/GradientDescent.hpp

        src/matrix/Matrix.hpp
        src/Network.hpp
        src/matrix/Algebraic.hpp
        src/activation/ActivationFactory.hpp
        src/matrix/3DTensor.hpp
        src/layers/Embedding.hpp
)

target_link_libraries(ML PRIVATE
        nlohmann_json::nlohmann_json
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)
target_include_directories(ML PRIVATE include)

# ----------------------------
# Examples
# ----------------------------

add_executable(example_mnist
        examples/mnist.cpp

        src/layers/Dense.hpp
        src/layers/Convolutional.hpp
        src/layers/Flatten.hpp
        src/layers/MaxPool2x2.hpp

        src/activation/Sigmoid.hpp
        src/lr_optimizer/Adam.hpp

        src/matrix/Matrix.hpp
        src/Network.hpp
        src/matrix/Algebraic.hpp
        src/activation/ActivationFactory.hpp
        src/matrix/3DTensor.hpp
)
target_link_libraries(example_mnist PRIVATE
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
)
target_include_directories(example_mnist SYSTEM PRIVATE ${stb_SOURCE_DIR})


# ----------------------------
# Copy Datasets to build directory
# ----------------------------

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/datasets/mnist)

file(COPY
        ${CMAKE_SOURCE_DIR}/datasets/mnist
        DESTINATION ${CMAKE_BINARY_DIR}/datasets
)