# cmake/KokkosKernelsConfig.cmake
# KokkosKernels options for Athelas.

set(KokkosKernels_INST_DOUBLE             ON  CACHE BOOL "")
set(KokkosKernels_INST_FLOAT              OFF CACHE BOOL "")
set(KokkosKernels_INST_COMPLEX_DOUBLE     OFF CACHE BOOL "")
set(KokkosKernels_INST_COMPLEX_FLOAT      OFF CACHE BOOL "")

# May need to be more careful abot layouts
set(KokkosKernels_INST_LAYOUTLEFT         OFF  CACHE BOOL "")
set(KokkosKernels_INST_LAYOUTRIGHT        ON CACHE BOOL "")

set(KokkosKernels_INST_ORDINAL_INT        ON  CACHE BOOL "")
set(KokkosKernels_INST_ORDINAL_INT64_T    OFF CACHE BOOL "")
set(KokkosKernels_INST_OFFSET_INT         ON  CACHE BOOL "")
set(KokkosKernels_INST_OFFSET_SIZE_T      OFF CACHE BOOL "")

set(KokkosKernels_ENABLE_ALL_COMPONENTS   OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON  CACHE BOOL "")

set(KokkosKernels_ENABLE_TPL_BLAS         OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_TPL_LAPACK       OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_TPL_CUBLAS       OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_TPL_CUSOLVER     OFF CACHE BOOL "")
set(KokkosKernels_ENABLE_TPL_CUSPARSE     OFF CACHE BOOL "")
