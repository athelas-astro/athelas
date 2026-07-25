# Originally based on code by Jonathan Hamberg
# https://gitlab.com/jhamberg/cmake-examples/-/tree/master/cmake
#
# Generates generated/build_info.cpp from src/build_info.cpp.in, stamping in the
# current git hash plus build provenance (compiler, optimization, arch, os,
# timestamp). The file is (re)written at configure time and refreshed before
# each build, so a new commit is reflected without a full reconfigure. A
# one-line hash cache (generated/git-state.txt) avoids needless relinks when the
# hash has not changed.
set(CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})

if (NOT DEFINED pre_configure_dir)
    set(pre_configure_dir ${CMAKE_SOURCE_DIR}/src)
endif ()

if (NOT DEFINED post_configure_dir)
    set(post_configure_dir ${CMAKE_SOURCE_DIR}/generated)
endif ()

set(pre_configure_file ${pre_configure_dir}/build_info.cpp.in)
set(post_configure_file ${post_configure_dir}/build_info.cpp)
set(git_state_file ${post_configure_dir}/git-state.txt)

# Provenance strings are only knowable during a normal CMake configure. In
# script mode (-P, the build-time refresh) they arrive via -D, so only derive
# them from the CMake variables when they were not already provided.
if (NOT DEFINED COMPILER)
    set(COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif ()
if (NOT DEFINED OPTIMIZATION)
    set(OPTIMIZATION "${CMAKE_BUILD_TYPE}")
endif ()
if (NOT DEFINED ARCH)
    set(ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif ()
if (NOT DEFINED OS)
    set(OS "${CMAKE_SYSTEM_NAME}")
endif ()

# (Re)write build_info.cpp with the current git hash and provenance. When
# `force` is false the write is skipped if the hash is unchanged and the file
# already exists, so a no-op build does not trigger a relink.
function(WriteBuildInfo force)
    execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CURRENT_LIST_DIR}
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        )
    if (NOT GIT_HASH)
        set(GIT_HASH "unknown")
    endif ()

    if (NOT EXISTS ${post_configure_dir})
        file(MAKE_DIRECTORY ${post_configure_dir})
    endif ()
    if (NOT EXISTS ${post_configure_dir}/build_info.hpp)
        file(COPY ${pre_configure_dir}/build_info.hpp
             DESTINATION ${post_configure_dir})
    endif ()

    set(cached_hash "")
    if (EXISTS ${git_state_file})
        file(READ ${git_state_file} cached_hash)
    endif ()

    if (force OR NOT GIT_HASH STREQUAL cached_hash OR
        NOT EXISTS ${post_configure_file})
        string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S %Z")
        configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
        file(WRITE ${git_state_file} ${GIT_HASH})
    endif ()
endfunction()

function(CheckGitSetup)
    # Refresh the hash before every build; the configure-time provenance
    # strings ride along via -D so the script-mode run reproduces them.
    add_custom_target(AthelasAlwaysCheckGit
        COMMAND ${CMAKE_COMMAND}
        -DRUN_WRITE_BUILD_INFO=1
        -Dpre_configure_dir=${pre_configure_dir}
        -Dpost_configure_dir=${post_configure_dir}
        "-DCOMPILER=${COMPILER}"
        "-DOPTIMIZATION=${OPTIMIZATION}"
        "-DARCH=${ARCH}"
        "-DOS=${OS}"
        -P ${CURRENT_LIST_DIR}/build_info.cmake
        BYPRODUCTS ${post_configure_file}
        )

    add_library(git_version ${post_configure_file})
    target_include_directories(git_version PUBLIC ${post_configure_dir})
    add_dependencies(git_version AthelasAlwaysCheckGit)

    # Always regenerate at configure time so a compiler/flag change is captured.
    WriteBuildInfo(TRUE)
endfunction()

# Script-mode entry point, invoked by the AthelasAlwaysCheckGit target each
# build. Respects the hash cache so unchanged builds do not relink.
if (RUN_WRITE_BUILD_INFO)
    WriteBuildInfo(FALSE)
endif ()
