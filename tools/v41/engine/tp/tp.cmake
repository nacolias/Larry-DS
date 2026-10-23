# tools/v41/engine/tp/tp.cmake - larry: engine1, the tensor-parallel decode core (libds41_tp.so) and its phase-A bench.
# Included from xe.cmake (inside its ggml-sycl block: the same AOT target; the shared decode kernels and fastmv come
# from ggml-sycl's sources, compiled here in their own namespaces).
set(_tp_dir ${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/engine/tp)
set(_tp_gsycl ${STRATA_GGML_DIR}/ggml/src/ggml-sycl)
add_library(ds41_tp SHARED ${_tp_dir}/tp_core.cpp ${_tp_dir}/tp_fastmv.cpp ${_tp_dir}/tp_pager.cpp)
target_include_directories(ds41_tp PUBLIC ${_tp_dir} PRIVATE ${_tp_gsycl} ${CMAKE_CURRENT_SOURCE_DIR}/include ${_v41_dir})
target_link_libraries(ds41_tp PRIVATE ds41_xe ggml-base ze_loader)   # ds41_xe: the planar expert kernels
target_compile_options(ds41_tp PRIVATE ${_xe_sycl})
target_link_options(ds41_tp PRIVATE ${_xe_sycl} -Wl,--exclude-libs,ALL)
add_executable(tp_bench ${_tp_dir}/tp_bench.cpp)
target_link_libraries(tp_bench PRIVATE ds41_tp)
target_compile_options(tp_bench PRIVATE -fsycl)
target_link_options(tp_bench PRIVATE -fsycl)
set_target_properties(tp_bench PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
# the engine: STRATA_TP=N runs its decode / verify passes on the core (ds41_generate.cpp, DS41_TP)
if(TARGET ds41_generate_xe)
    target_link_libraries(ds41_generate_xe PRIVATE ds41_tp)
    target_compile_definitions(ds41_generate_xe PRIVATE DS41_TP)
endif()
