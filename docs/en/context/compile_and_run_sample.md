# Compile and Run Samples

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T22:50:48.713Z -->

## Prerequisites

- To compile and execute operator APIs, ensure that the basic environment is set up, including the driver, firmware, CANN software package, and ops package.
- For details about the operator API invocation process and compilation and running operations, see "Single Operator Invocation > Single Operator API Execution > Example Code for Calling aclnn APIs" in [*App Development (C&C++)*](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/aclcppdevg_000006.html).

## Preparations Before Compilation

This chapter uses as an example the case where the development and running environments are co-located, that is, the machine with an AI processor serves as both the development environment and the running environment. In this case, code development and code running are performed on the same machine. The **Abs operator** is used as an example here. The invocation logic, process, and build script of other operators are similar to those of the Abs operator. Modify the API invocation script (\*.cpp) and the build script (CMakeLists) as required.

- **Sample Code**

   The Abs operator computes the absolute value of each element in a tensor, with the formula: y<sub>i</sub>=|x<sub>i</sub>|. You can obtain the sample invocation code from the aclnnAbs documentation and name the code file **test\_abs.cpp**.

- **CMakeLists File**

    The following is an example of the CMake file. Modify it based on the actual situation:

    ```cmake
    # Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set the project name.
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set the compilation options.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set the executable file name (for example, opapi_test) and specify the directory where the *.cpp file of the operator to be run is located.
    add_executable(opapi_test
                   test_abs.cpp)

    # Set ASCEND_PATH (directory of the CANN software package, modify it based on the actual path) and INCLUDE_BASE_DIR (directory of header files).
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # Set the paths of the libraries to be linked.
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so)

    # The executable file is stored in the bin directory under the directory where the CMakeLists file resides.
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    Operators that integrate collective communication with MatMul computation and parallelism are collectively referred to as communication-computation fusion operators (MC2 operators for short), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, MatmulReduceScatter, and others. When calling APIs of such operators, multi-threading and HCCL (Huawei Collective Communication Library) are typically involved. Therefore, the CMake file must additionally import the following content; otherwise, compilation will fail.

  ```cmake
  # Set the paths of the libraries to be linked.
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${CMAKE_THREAD_LIBS_INIT})           # Library file on which multithreading depends
  ```

  Here, "find_package(Threads REQUIRED)" is a CMake command for locating the thread library, which automatically links the header files that the thread library depends on or the indirectly dependent library files.

## Compile and Run

  1. Prepare the operator invocation code (\*.cpp) and the build script (CMakeLists.txt) in advance.
  2. Configure environment variables.

     After installing the CANN software, log in to the CANN runtime environment as the user and execute the following command to make the environment variables take effect.

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     Where ${INSTALL_DIR} is the storage path of the CANN software after installation. Replace it with the actual path as needed.

  3. Compile and run.

      - Go to the directory where CMakeLists.txt is located and run the following command to create a build directory for storing the generated compilation files.

          ```bash
          mkdir -p build
          ```

      - Go to the build directory, run the cmake command to compile, and then run the make command to generate the executable file.

        ```bash
        cd build
        cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
        make
        ```

        After successful compilation, the executable file opapi_test is generated in the bin folder under the build directory.

      - Go to the bin directory and run the executable file opapi_test.

        ```bash
        cd bin
        ./opapi_test
        ```

        Taking the running result of the Abs operator as an example, the result after running is as follows:

        ```bash
        result[0] is: 1.000000
        result[1] is: 1.000000
        result[2] is: 1.000000
        result[3] is: 2.000000
        result[4] is: 2.000000
        result[5] is: 2.000000
        result[6] is: 3.000000
        result[7] is: 3.000000
        ```

        If an error is reported and the expected result is not obtained, call the aclGetRecentErrMsg API to obtain detailed error information.
        The following is an example of obtaining error information when a call to aclnnAbsGetWorkspaceSize fails:

        ```bash
        // self is nullptr
        ret = aclnnAbsGetWorkspaceSize(self, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAbsGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
        ```

        The following is an example of obtaining error information for the preceding null pointer construction problem:

        ```bash
        aclnnAbsGetWorkspaceSize failed. ERROR: 161001
        [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): Expected a value of type [aclTensor] for argument [self] but instead found nullptr.
        ```
