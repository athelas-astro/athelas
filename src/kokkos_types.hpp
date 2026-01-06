#pragma once

#include <Kokkos_Core.hpp>

namespace athelas {
#ifdef KOKKOS_ENABLE_CUDA_UVM
using DevMemSpace = Kokkos::CudaUVMSpace;
using HostMemSpace = Kokkos::CudaUVMSpace;
using DevExecSpace = Kokkos::Cuda;
#else
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using DevExecSpace = Kokkos::DefaultExecutionSpace;
#endif
using ScratchMemSpace = DevExecSpace::scratch_memory_space;

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using LayoutWrapper = Kokkos::LayoutRight;
using MemUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

// AthelasArrays (that directly map a view)
template <typename T>
using AthelasArray0D = Kokkos::View<T, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthelasArray1D = Kokkos::View<T *, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthelasArray2D = Kokkos::View<T **, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthelasArray3D = Kokkos::View<T ***, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthelasArray4D = Kokkos::View<T ****, LayoutWrapper, DevMemSpace>;
template <typename T>
using AthelasArray5D = Kokkos::View<T *****, LayoutWrapper, DevMemSpace>;

// Host mirrors
template <typename T>
using HostArray0D = typename AthelasArray0D<T>::HostMirror;
template <typename T>
using HostArray1D = typename AthelasArray1D<T>::HostMirror;
template <typename T>
using HostArray2D = typename AthelasArray2D<T>::HostMirror;
template <typename T>
using HostArray3D = typename AthelasArray3D<T>::HostMirror;
template <typename T>
using HostArray4D = typename AthelasArray4D<T>::HostMirror;
template <typename T>
using HostArray5D = typename AthelasArray5D<T>::HostMirror;

using team_policy = Kokkos::TeamPolicy<>;
using team_mbr_t = Kokkos::TeamPolicy<>::member_type;

template <typename T>
using ScratchPad1D =
    Kokkos::View<T *, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad2D =
    Kokkos::View<T **, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
} // namespace athelas
