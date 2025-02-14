set(SOURCE_FILES
    StepwiseGT.cpp
    StepwiseRecall.cpp
)

add_executable(ComputeStepwiseGT ${SOURCE_FILES})
target_link_libraries(ComputeStepwiseGT PRIVATE Boost::program_options OpenMP::OpenMP_CXX mkl_rt)
