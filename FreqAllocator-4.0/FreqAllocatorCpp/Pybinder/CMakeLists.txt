cmake_minimum_required(VERSION 3.1)
project(FreqAllocatorCpp)

# set(LIB_FreqAllocator ${PROJECT_NAME})
file(GLOB_RECURSE FreqAllocatorWrapper_SRC_CPP *.cpp)

pybind11_add_module(${PROJECT_NAME} ${FreqAllocatorWrapper_SRC_CPP})

target_link_libraries(${PROJECT_NAME} PRIVATE FreqAllocatorCore fmt)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${FREQ_ALLOCATOR_VERSION_INFO})