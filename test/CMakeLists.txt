cmake_minimum_required(VERSION 3.14)

project(rabitq_testqg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

set(RABITQ_TEST_INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${CMAKE_CURRENT_LIST_DIR}
)

add_executable(testqg testqg.cpp)
target_include_directories(testqg PRIVATE ${RABITQ_TEST_INCLUDES})
target_compile_options(testqg PRIVATE -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl)
if (OpenMP_CXX_FOUND)
  target_link_libraries(testqg PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(saveqg saveqg.cpp)
target_include_directories(saveqg PRIVATE ${RABITQ_TEST_INCLUDES})
target_compile_options(saveqg PRIVATE -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl)
if (OpenMP_CXX_FOUND)
  target_link_libraries(saveqg PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(load_test_qg load_test_qg.cpp)
target_include_directories(load_test_qg PRIVATE ${RABITQ_TEST_INCLUDES})
target_compile_options(load_test_qg PRIVATE -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl)
if (OpenMP_CXX_FOUND)
  target_link_libraries(load_test_qg PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(load_test_qg_concurrently load_test_qg_concurrently.cpp)
target_include_directories(load_test_qg_concurrently PRIVATE ${RABITQ_TEST_INCLUDES})
target_compile_options(load_test_qg_concurrently PRIVATE -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl)
if (OpenMP_CXX_FOUND)
  target_link_libraries(load_test_qg_concurrently PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(print_layout print_layout.cpp)
target_include_directories(print_layout PRIVATE ${RABITQ_TEST_INCLUDES})
target_compile_options(print_layout PRIVATE -mavx2 -mfma -mavx512f -mavx512dq -mavx512bw -mavx512vl)
