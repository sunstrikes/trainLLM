cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_C_FLAGS "-O3 -g -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -lm")
SET(CMAKE_CXX_FLAGS "-O3 -g -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -DEIGEN_NO_DEBUG")

project(llm.cpp)

aux_source_directory(./dev/src DIR_SRCS)

add_executable(train_gpt train_gpt2.c)

add_executable(cpu_train_gpt ${DIR_SRCS})
#add_test (test_2_10 Demo 2 10)

#set_tests_properties (test_2_10
#        PROPERTIES PASS_REGULAR_EXPRESSION "is 1024")
