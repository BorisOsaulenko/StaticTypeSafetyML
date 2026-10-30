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
        tests/layers/embedding_test.cpp
        tests/layers/attention_head_test.cpp
        tests/layers/add_and_norm_test.cpp

        tests/network/network_test.cpp
        tests/lr_optimizer/adam_test.cpp
        tests/linear_regression_test.cpp

        src/layers/Embedding.h
        src/layers/ParamsLayer.h
        src/layers/Flatten.h
        src/layers/AttentionHead.h

        src/loss/Loss.h
        src/loss/MSE.h
        src/loss/LossFactory.h

        src/lr_optimizer/Adam.h
        src/lr_optimizer/Optimizer.h

        src/matrix/Algebraic.h
        src/matrix/StaticArray.h
        src/preprocessing/tokenizer.h

        tests/utils.h
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

        src/layers/NoParamsLayer.h
        src/layers/Dense.h
        src/layers/Convolutional.h
        src/layers/Flatten.h
        src/layers/MaxPool2x2.h

        src/activation/ActivationFunction.h

        src/activation/Sigmoid.h
        src/activation/ReLu.h
        src/activation/Identity.h

        src/lr_optimizer/Optimizer.h
        src/lr_optimizer/Adam.h
        src/lr_optimizer/GradientDescent.h

        src/matrix/Matrix.h
        src/Network.h
        src/matrix/Algebraic.h
        src/activation/ActivationFactory.h
        src/matrix/3DTensor.h
        src/layers/Embedding.h
)

# ----------------------------
# Link libraries
# ----------------------------

target_include_directories(ML PRIVATE include)

target_link_libraries(ML PRIVATE
        nlohmann_json::nlohmann_json
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)

target_include_directories(ML PRIVATE ${OpenCV_INCLUDE_DIRS})

# ----------------------------
# Copy Datasets to build directory
# ----------------------------

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/datasets/mnist)

file(COPY
        ${CMAKE_SOURCE_DIR}/datasets/mnist
        DESTINATION ${CMAKE_BINARY_DIR}/datasets
)