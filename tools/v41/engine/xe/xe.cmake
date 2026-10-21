# tools/v41/engine/xe/xe.cmake - larry: the DeepSeek-V4.1 engine with the multi-card Intel Arc (SYCL) expert tier.
# Needs ggml's SYCL backend (-DGGML_SYCL=ON, icpx) and -DCMAKE_POSITION_INDEPENDENT_CODE=ON (the tier is a shared
# library: its own AOT device code, so changing it never re-links ggml-sycl's device images into the engine - that
# link alone takes ~10 minutes).  Included from tools/v41/engine/v41_engine.cmake.
if(TARGET ggml-sycl AND TARGET ds41_dense AND TARGET ds41_moe_cpu)
    set(_xe_dir ${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/engine/xe)
    set(STRATA_XE_AOT "bmg-g31" CACHE STRING "Intel GPU AOT device(s) for the Xe expert kernels (empty = JIT)")
    if(STRATA_XE_AOT)
        set(_xe_sycl -fsycl -fsycl-targets=spir64_gen "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${STRATA_XE_AOT}\"")
    else()
        set(_xe_sycl -fsycl)
    endif()
    add_library(ds41_xe SHARED ${_xe_dir}/xe_kernels.cpp ${_xe_dir}/xe_planar.cpp ${_xe_dir}/xe_xmx.cpp ${_xe_dir}/xe_dqgemm.cpp ${_xe_dir}/ds41_xe_tier.cpp)
    target_include_directories(ds41_xe PUBLIC ${_xe_dir} ${CMAKE_CURRENT_SOURCE_DIR}/include ${_v41_dir})
    find_package(MKL CONFIG REQUIRED)   # oneMKL SYCL BLAS: the prompt path's fp16 GEMMs
    target_link_libraries(ds41_xe PRIVATE ds41_moe_cpu strata_kernels_cpu ggml-cpu ggml-base MKL::MKL_SYCL::BLAS ze_loader)
    target_compile_options(ds41_xe PRIVATE ${_xe_sycl})
    # its private copies of ggml / the CPU kernels stay private: the engine must resolve ggml from its OWN static copy
    target_link_options(ds41_xe PRIVATE ${_xe_sycl} -Wl,--exclude-libs,ALL)
    add_executable(ds41_generate_xe ${_v41_dir}/ds41_generate.cpp ${_v41_dir}/ds41_dspark.cpp)
    target_compile_definitions(ds41_generate_xe PRIVATE DS41_XE DS41_HAVE_DSPARK $<$<BOOL:${STRATA_DSV41_FQ_OP}>:STRATA_DSV41_FQ_OP>)
    target_link_libraries(ds41_generate_xe PRIVATE ds41_dense ds41_moe_cpu ds41_xe)
    # ggml-sycl's objects carry spir64_gen (AOT) device code: the engine links with the same target
    # the device link of ggml-sycl's kernels: one AOT job per source, 16 at a time (~2 min instead of ~10)
    set(_xe_dlink -fsycl-device-code-split=per_source -fsycl-max-parallel-link-jobs=16)
    target_link_options(ds41_generate_xe PRIVATE ${_xe_sycl} ${_xe_dlink})
    # a new libds41_xe.so does not re-link the engine (its ABI is the header, which the engine's object depends on)
    set_target_properties(ds41_generate_xe PROPERTIES LINK_DEPENDS_NO_SHARED ON BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
    # every other executable that links ggml-sycl statically needs the same device link (else its kernels are missing)
    foreach(_t ds41_generate test_ds41_fq test_ds41_hcmix test_ds41_hcfuse test_ds41_ops test_ds41_kvfp4 test_ds41_dspark)
        if(TARGET ${_t})
            target_link_options(${_t} PRIVATE ${_xe_sycl})
        endif()
    endforeach()
    add_executable(test_ds41_xe ${_xe_dir}/test_ds41_xe.cpp)
    target_link_libraries(test_ds41_xe PRIVATE ds41_xe ds41_moe_cpu strata_kernels_cpu ggml-cpu ggml-base MKL::MKL_SYCL::BLAS)
    target_compile_options(test_ds41_xe PRIVATE -fsycl)
    target_link_options(test_ds41_xe PRIVATE ${_xe_sycl})
endif()
