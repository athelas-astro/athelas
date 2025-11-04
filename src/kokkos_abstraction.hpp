#pragma once

#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include "basic_types.hpp"
#include "kokkos_types.hpp"
#include "loop_bounds.hpp"
#include "loop_layout.hpp"
#include "utils/indexer.hpp"
#include "utils/type_list.hpp"

/**
 * The code here is largely borrowed from Parthenon
 * https://github.com/parthenon-hpc-lab/parthenon
 * And contains wrappers around Kokkos parallel regions allowing
 * 1) a nicer API for writing parallel regions and
 * 2) compile time switching for looping patterns.
 */

namespace athelas {

// Defining tags to determine loop_patterns using a tag dispatch design pattern

// Translates to a non-Kokkos standard C++ nested `for` loop where the innermost
// `for` is decorated with a #pragma omp simd IMPORTANT: This only works on CPUs
static const struct LoopPatternSimdFor {
} loop_pattern_simdfor_tag;
// Translates to a Kokkos 1D range (Kokkos::RangePolicy) where the wrapper takes
// care of the (hidden) 1D index to `n`, `k`, `j`, `i indices conversion
static const struct LoopPatternFlatRange {
} loop_pattern_flatrange_tag;
// Translates to a Kokkos multi dimensional  range (Kokkos::MDRangePolicy) with
// a 1:1 indices matching
static const struct LoopPatternMDRange {
} loop_pattern_mdrange_tag;
// Translates to a Kokkos::TeamPolicy that collapse Nthread & Nvector inner
// loops
template <std::size_t num_thread, std::size_t num_vector>
struct LoopPatternTeamThreadVec {
  static constexpr std::size_t Nthread = num_thread;
  static constexpr std::size_t Nvector = num_vector;
};

// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::TeamThreadRange
using LoopPatternTPTTR = LoopPatternTeamThreadVec<1, 0>;
constexpr auto loop_pattern_tpttr_tag = LoopPatternTPTTR();
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::ThreadVectorRange
using LoopPatternTPTVR = LoopPatternTeamThreadVec<0, 1>;
constexpr auto loop_pattern_tptvr_tag = LoopPatternTPTVR();
// Translates to a Kokkos::TeamPolicy with a middle Kokkos::TeamThreadRange and
// inner Kokkos::ThreadVectorRange
using LoopPatternTPTTRTVR = LoopPatternTeamThreadVec<1, 1>;
constexpr auto loop_pattern_tpttrtvr_tag = LoopPatternTPTTRTVR();
// Translates to an outer team policy
using LoopPatternTeamOuter = LoopPatternTeamThreadVec<0, 0>;
constexpr auto loop_pattern_team_outer_tag = LoopPatternTeamOuter();
// Used to catch undefined behavior as it results in throwing an error
static const struct LoopPatternUndefined {
} loop_pattern_undefined_tag;

// Tags for Nested parallelism

// Translates to outermost loop being a Kokkos::TeamPolicy for par_for_outer
// like loops
static const struct OuterLoopPatternTeams {
} outer_loop_pattern_teams_tag;

// collapses Nvector inner loops over a VectorRange policy and remaining over a
// ThreadRange
template <std::size_t Nvector>
struct InnerLoopThreadVec {};

// Inner loop pattern tags must be constexpr so they're available on device
// Translate to a Kokkos::TeamVectorRange as innermost loop (single index)
using InnerLoopPatternTVR = InnerLoopThreadVec<1>;
constexpr InnerLoopPatternTVR inner_loop_pattern_tvr_tag;
// Translates to a Kokkos::TeamThreadRange as innermost loop
using InnerLoopPatternTTR = InnerLoopThreadVec<0>;
constexpr InnerLoopPatternTTR inner_loop_pattern_ttr_tag;
// Translate to a non-Kokkos plain C++ innermost loop (single index)
// decorated with #pragma omp simd
// IMPORTANT: currently only supported on CPUs
struct InnerLoopPatternSimdFor {};
constexpr InnerLoopPatternSimdFor inner_loop_pattern_simdfor_tag;

// trait to track if pattern requests any type of hierarchial parallelism
template <typename Pattern, typename T = void>
struct UsesHierarchicalPar : std::false_type {
  static constexpr std::size_t Nvector = 0;
  static constexpr std::size_t Nthread = 0;
};

template <std::size_t num_thread, std::size_t num_vector>
struct UsesHierarchicalPar<LoopPatternTeamThreadVec<num_thread, num_vector>>
    : std::true_type {
  static constexpr std::size_t Nthread = num_thread;
  static constexpr std::size_t Nvector = num_vector;
};

template <>
struct UsesHierarchicalPar<OuterLoopPatternTeams> : std::true_type {
  static constexpr std::size_t Nvector = 0;
  static constexpr std::size_t Nthread = 0;
};

template <std::size_t num_vector>
struct UsesHierarchicalPar<InnerLoopThreadVec<num_vector>> : std::true_type {
  static constexpr std::size_t Nvector = num_vector;
};

namespace dispatch_impl {
static const struct ParallelForDispatch {
} parallel_for_dispatch_tag;
static const struct ParallelReduceDispatch {
} parallel_reduce_dispatch_tag;
static const struct ParallelScanDispatch {
} parallel_scan_dispatch_tag;

template <class... Args>
inline void kokkos_dispatch(ParallelForDispatch, Args &&...args) {
  Kokkos::parallel_for(std::forward<Args>(args)...);
}
template <class... Args>
inline void kokkos_dispatch(ParallelReduceDispatch, Args &&...args) {
  Kokkos::parallel_reduce(std::forward<Args>(args)...);
}
template <class... Args>
inline void kokkos_dispatch(ParallelScanDispatch, Args &&...args) {
  Kokkos::parallel_scan(std::forward<Args>(args)...);
}
} // namespace dispatch_impl

template <typename>
struct DispatchSignature {};

template <typename... AllArgs>
struct DispatchSignature<TypeList<AllArgs...>> {
 private:
  using TL = TypeList<AllArgs...>;
  static constexpr std::size_t func_idx = FirstFuncIdx<TL>();
  static_assert(sizeof...(AllArgs) > func_idx,
                "Couldn't determine functor index from dispatch args");

 public:
  using LoopBounds = typename TL::template continuous_sublist<0, func_idx - 1>;
  using Translator = LoopBoundTranslator<LoopBounds>;
  static constexpr std::size_t Rank = Translator::Rank;
  using Function = typename TL::template type<func_idx>;
  using Args = typename TL::template continuous_sublist<func_idx + 1>;
};

// tags to resolve requested [Outer|Inner]LoopPattern* +
// dispatch_impl::Parallel*Dispatch combinations into the final
// ParDispatchImpl::dispatch_impl call using DispatchType::GetPatternTag() to
// prevent any incompatible combinations (e.g., simdfor + par_reduce).
// * flat     -- results in a single Kokkos::RangePolicy flattening all loop
// bounds.
// * md       -- results in a Kokkos::MDRangePolicy
// * simd     -- innermost loop gets a #pragma omp simd, outer loops flattened
// to a single
//               raw for
// * outer    -- only explicit parallelism is an outer team_policy
// * collapse -- Any specialization of [Inner]LoopPatternCollapse patterns.
// Explicitly
//               uses hierarchial parrallelism
// * undef    -- combination not handled, will raise a compilation error.
enum class PatternTag { flat, md, simd, outer, collapse, undef };

template <PatternTag PTag>
struct LoopPatternTag {};

template <typename Tag, typename Pattern, typename... Bounds>
struct DispatchType {
  using Translator = LoopBoundTranslator<Bounds...>;
  static constexpr std::size_t Rank = Translator::Rank;

  using HierarchicalPar = UsesHierarchicalPar<Pattern>;

  static constexpr bool is_ParFor =
      std::is_same_v<Tag, dispatch_impl::ParallelForDispatch>;
  static constexpr bool is_ParRed =
      std::is_same_v<Tag, dispatch_impl::ParallelReduceDispatch>;
  static constexpr bool is_ParScan =
      std::is_same_v<Tag, dispatch_impl::ParallelScanDispatch>;

  static constexpr bool IsFlatRange =
      std::is_same_v<Pattern, LoopPatternFlatRange>;
  static constexpr bool IsMDRange = std::is_same_v<Pattern, LoopPatternMDRange>;
  static constexpr bool IsSimdFor = std::is_same_v<Pattern, LoopPatternSimdFor>;
  static constexpr bool IsSimdForInner =
      std::is_same_v<Pattern, InnerLoopPatternSimdFor>;

  // check any confilcts with the requested pattern
  // and return the actual one we use
  static constexpr PatternTag GetPatternTag() {
    using PT = PatternTag;

    if constexpr (is_ParScan) {
      return PT::flat;
    } else if constexpr (IsFlatRange) {
      return PT::flat;
    } else if constexpr (IsSimdFor) {
      return is_ParFor ? PT::simd : PT::flat;
    } else if constexpr (IsSimdForInner) {
      // for now this is guaranteed to be par_for_inner, when par_reduce_inner
      // is supported need to check
      return PT::simd;
    } else if constexpr (IsMDRange || is_ParRed) {
      // par_reduce does not currently work with either team-based patterns
      return PT::md;
    } else if constexpr (std::is_same_v<Pattern, OuterLoopPatternTeams>) {
      return PT::outer;
    } else if constexpr (HierarchicalPar::value) {
      return PT::collapse;
    }

    return PT::undef;
  }
};

template <std::size_t Rank, typename IdxTeam, std::size_t Nteam,
          std::size_t Nthread, std::size_t Nvector, typename Function,
          typename... ExtraFuncArgs>
struct DispatchCollapse {
  IdxTeam idxer_team;
  Kokkos::Array<IndexRange, Rank> bound_arr;
  Function function;
  using PT = PatternTag;

  KOKKOS_FORCEINLINE_FUNCTION
  DispatchCollapse(IdxTeam idxer, Kokkos::Array<IndexRange, Rank> bounds,
                   Function func)
      : idxer_team(idxer), bound_arr(bounds), function(func) {}

  // collapse inner parallel regions using a combination of Team/Thread/Vector
  // range policies
  template <std::size_t... TeamIs, std::size_t... ThreadIs,
            std::size_t... VectorIs, typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION void
  execute(LoopPatternTag<PT::collapse>,
          std::integer_sequence<std::size_t, TeamIs...>,
          std::integer_sequence<std::size_t, ThreadIs...>,
          std::integer_sequence<std::size_t, VectorIs...>,
          team_mbr_t team_member, Args &&...args) const {
    auto inds_team = idxer_team.get_idx_array(team_member.league_rank());
    if constexpr (Nthread > 0) {
      const auto idxer_thread = make_indexer(
          Kokkos::Array<IndexRange, Nthread>{bound_arr[ThreadIs]...});
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member, 0, idxer_thread.size()),
          [&](const int idThread, ExtraFuncArgs... fargs) {
            const auto inds_thread = idxer_thread.get_idx_array(idThread);
            if constexpr (Nvector > 0) {
              static_assert(
                  Nvector * Nthread == 0 || sizeof...(Args) == 0,
                  "thread + vector range pattern only supported for par_for ");
              const auto idxer_vector =
                  make_indexer(Kokkos::Array<IndexRange, Nvector>{
                      bound_arr[Nthread + VectorIs]...});
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, 0,
                                            idxer_vector.size()),
                  [&](const int idVector) {
                    const auto inds_vector =
                        idxer_vector.get_idx_array(idVector);
                    function(inds_team[TeamIs]..., inds_thread[ThreadIs]...,
                             inds_vector[VectorIs]...,
                             std::forward<ExtraFuncArgs>(fargs)...);
                  });
            } else {
              function(inds_team[TeamIs]..., inds_thread[ThreadIs]...,
                       std::forward<ExtraFuncArgs>(fargs)...);
            }
          },
          std::forward<Args>(args)...);
    } else {
      const auto idxer_vector = make_indexer(
          Kokkos::Array<IndexRange, Nvector>{bound_arr[Nthread + VectorIs]...});
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(team_member, 0, idxer_vector.size()),
          [&](const int idVector, ExtraFuncArgs... fargs) {
            const auto inds_vector = idxer_vector.get_idx_array(idVector);
            function(inds_team[TeamIs]..., inds_vector[VectorIs]...,
                     std::forward<ExtraFuncArgs>(fargs)...);
          },
          std::forward<Args>(args)...);
    }
  }

  // simdfor loop collapse inside an outer team policy loop. Only valid on
  // HostExecSpace
  template <std::size_t... OuterIs, std::size_t... ThreadIs,
            std::size_t... InnerIs, typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION void
  execute(LoopPatternTag<PT::simd>,
          std::integer_sequence<std::size_t, OuterIs...>,
          std::integer_sequence<std::size_t, ThreadIs...>,
          std::integer_sequence<std::size_t, InnerIs...>,
          team_mbr_t team_member, Args &&...args) const {
    static_assert(sizeof...(ThreadIs) == Rank - 1);
    static_assert(sizeof...(OuterIs) == 0 && sizeof...(InnerIs) == 0,
                  "simd inner pattern should only provide thread indices");
    if constexpr (Rank == 1) {
#pragma omp simd
      for (int i = bound_arr[0].s; i <= bound_arr[0].e; i++) {
        function(i);
      }
    } else {
      const auto idxer = make_indexer(
          std::pair<int, int>(bound_arr[ThreadIs].s, bound_arr[ThreadIs].e)...);
      for (int idx = 0; idx < idxer.size(); idx++) {
        const auto indices = idxer.get_idx_array(idx);
#pragma omp simd
        for (int i = bound_arr[Rank - 1].s; i <= bound_arr[Rank - 1].e; i++) {
          function(indices[ThreadIs]..., i);
        }
      }
    }
  }

  template <std::size_t N>
  using sequence = std::make_index_sequence<N>;
  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(team_mbr_t team_member) const {
    execute(LoopPatternTag<PT::collapse>(), sequence<Nteam>(),
            sequence<Nthread>(), sequence<Nvector>(), team_member);
  }
};

// builds a functor that uses inner hierarchial parrallelism used by both
// par_disp_inner & par_dispatch for LoopPatternCollapse
template <std::size_t Rank, std::size_t Nteam, std::size_t Nthread,
          std::size_t Nvector, typename IdxTeam, typename Function,
          typename... ExtraFuncArgs>
KOKKOS_FORCEINLINE_FUNCTION auto
make_collapse(IdxTeam idxer, Kokkos::Array<IndexRange, Rank> bounds,
              Function func) {
  return DispatchCollapse<Rank, IdxTeam, Nteam, Nthread, Nvector, Function,
                          ExtraFuncArgs...>(idxer, bounds, func);
}

template <typename, typename, typename, typename, typename>
struct ParDispInnerImpl {};

template <typename Pattern, typename Function, typename... Bounds,
          typename... Args, typename... ExtraFuncArgs>
struct ParDispInnerImpl<Pattern, Function, TypeList<Bounds...>,
                        TypeList<Args...>, TypeList<ExtraFuncArgs...>> {
  using bound_translator = LoopBoundTranslator<Bounds...>;
  using dispatch_type =
      DispatchType<dispatch_impl::ParallelForDispatch, Pattern, Bounds...>;
  static constexpr std::size_t Rank = bound_translator::Rank;

  template <std::size_t N>
  using sequence = std::make_index_sequence<N>;

  KOKKOS_FORCEINLINE_FUNCTION void execute(team_mbr_t team_member,
                                           Bounds &&...bounds,
                                           Function function, Args &&...args) {
    auto bound_arr =
        bound_translator().GetIndexRanges(std::forward<Bounds>(bounds)...);
    constexpr bool isSimdFor = std::is_same_v<InnerLoopPatternSimdFor, Pattern>;
    constexpr std::size_t Nvector = dispatch_type::HierarchicalPar::Nvector;
    constexpr std::size_t Nthread = Rank - Nvector;
    constexpr auto pattern_tag =
        LoopPatternTag<dispatch_type::GetPatternTag()>();

    static_assert(!isSimdFor || (isSimdFor &&
                                 std::is_same_v<DevExecSpace, HostExecSpace>),
                  "par_inner simd for pattern only supported on HostExecSpace");
    static_assert(
        !std::is_same_v<decltype(pattern_tag),
                        LoopPatternTag<PatternTag::undef>> ||
            always_false<Pattern>,
        "Inner Loop pattern not recognized in DispatchType::GetPatternTag");

    auto idxer = Indexer<>();
    make_collapse<Rank, 0, Nthread, Nvector, ExtraFuncArgs...>(idxer, bound_arr,
                                                               function)
        .execute(pattern_tag, sequence<0>(), sequence<Nthread - isSimdFor>(),
                 sequence<Nvector>(), team_member, std::forward<Args>(args)...);
  }
};

template <typename Pattern, typename... AllArgs>
KOKKOS_FORCEINLINE_FUNCTION void par_disp_inner(Pattern, team_mbr_t team_member,
                                                AllArgs &&...args) {
  using dispatchsig = DispatchSignature<TypeList<AllArgs...>>;
  constexpr std::size_t Rank = dispatchsig::Rank;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;
  using Args = typename dispatchsig::Args;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;
  ParDispInnerImpl<Pattern, Function, LoopBounds, Args, ExtraFuncArgs>()
      .execute(team_member, std::forward<AllArgs>(args)...);
}

template <typename, typename, typename, typename, typename, typename>
struct ParDispatchImpl {};

template <typename Tag, typename Pattern, typename Function, typename... Bounds,
          typename... Args, typename... ExtraFuncArgs>
struct ParDispatchImpl<Tag, Pattern, Function, TypeList<Bounds...>,
                       TypeList<Args...>, TypeList<ExtraFuncArgs...>> {
  using PT = PatternTag;
  using dispatch_type = DispatchType<Tag, Pattern, Bounds...>;
  using bound_translator = LoopBoundTranslator<Bounds...>;
  static constexpr std::size_t Rank = bound_translator::Rank;

  template <typename ExecSpace>
  inline void dispatch(std::string name, ExecSpace exec_space,
                       Bounds &&...bounds, Function function, Args &&...args,
                       const int scratch_level = 0,
                       const std::size_t scratch_size_in_bytes = 0) {
    constexpr std::size_t Ninner = dispatch_type::HierarchicalPar::Nvector +
                                   dispatch_type::HierarchicalPar::Nthread;

    constexpr auto pattern_tag =
        LoopPatternTag<dispatch_type::GetPatternTag()>();
    static_assert(
        !std::is_same_v<decltype(pattern_tag), LoopPatternTag<PT::undef>> &&
            !always_false<Tag, Pattern>,
        "LoopPattern & Tag combination not recognized in "
        "DispatchType::GetPatternTag");

    constexpr bool isSimdFor = std::is_same_v<LoopPatternTag<PatternTag::simd>,
                                              base_type<decltype(pattern_tag)>>;
    static_assert(!isSimdFor ||
                      (isSimdFor && std::is_same_v<ExecSpace, HostExecSpace>),
                  "SimdFor pattern only supported in HostExecSpace");

    auto bound_arr =
        bound_translator().GetIndexRanges(std::forward<Bounds>(bounds)...);
    dispatch_impl(pattern_tag,
                  std::make_index_sequence<Rank - Ninner - isSimdFor>(),
                  std::make_index_sequence<Ninner>(), name, exec_space,
                  bound_arr, function, std::forward<Args>(args)...,
                  scratch_level, scratch_size_in_bytes);
  }

  template <std::size_t... Is>
  using sequence = std::integer_sequence<std::size_t, Is...>;

  // #pragma omp simd for innermost loop, flatten remaining outer loops into a
  // single raw for
  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void
  dispatch_impl(LoopPatternTag<PT::simd>, sequence<OuterIs...>,
                sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                Args &&...args, const int scratch_level,
                const std::size_t scratch_size_in_bytes) {
    static_assert(sizeof...(InnerIs) == 0);
    static_assert(sizeof...(OuterIs) == Rank - 1);
    if constexpr (Rank == 1) {
#pragma omp simd
      for (int i = bound_arr[0].s; i <= bound_arr[0].e; i++) {
        function(i);
      }
    } else {
      const auto idxer = make_indexer(
          std::pair<int, int>(bound_arr[OuterIs].s, bound_arr[OuterIs].e)...);
      for (std::size_t idx = 0; idx < idxer.size(); idx++) {
        const auto indices = idxer.get_idx_array(idx);
#pragma omp simd
        for (int i = bound_arr[Rank - 1].s; i <= bound_arr[Rank - 1].e; i++) {
          function(indices[OuterIs]..., i);
        }
      }
    }
  }

  // flatten all loop bounds onto a single Kokkos::RangePolicy
  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void
  dispatch_impl(LoopPatternTag<PT::flat>, sequence<OuterIs...>,
                sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                Args &&...args, const int scratch_level,
                const std::size_t scratch_size_in_bytes) {
    static_assert(sizeof...(InnerIs) == 0);
    const auto idxer = make_indexer(bound_arr);
    kokkos_dispatch(
        Tag(), name, Kokkos::RangePolicy<>(exec_space, 0, idxer.size()),
        KOKKOS_LAMBDA(const int idx, ExtraFuncArgs... fargs) {
          const auto idx_arr = idxer.get_idx_array(idx);
          function(idx_arr[OuterIs]..., std::forward<ExtraFuncArgs>(fargs)...);
        },
        std::forward<Args>(args)...);
  }

  // Kokkos::MDRangePolicy for all loop bounds
  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void
  dispatch_impl(LoopPatternTag<PT::md>, sequence<OuterIs...>,
                sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                Args &&...args, const int scratch_level,
                const std::size_t scratch_size_in_bytes) {
    static_assert(sizeof...(InnerIs) == 0);
    constexpr std::size_t Nouter = sizeof...(OuterIs);
    Kokkos::Array<int, Nouter> tiling;
    for (std::size_t i = 0; i < Nouter - 1; i++)
      tiling[i] = 1;
    tiling[Nouter - 1] = bound_arr[Nouter - 1].e + 1 - bound_arr[Nouter - 1].s;
    kokkos_dispatch(
        Tag(), name,
        Kokkos::Experimental::require(
            Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(
                exec_space, {bound_arr[OuterIs].s...},
                {(1 + bound_arr[OuterIs].e)...}, tiling),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        function, std::forward<Args>(args)...);
  }

  // Flatten loop bounds into a single outer team_policy
  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void
  dispatch_impl(LoopPatternTag<PT::outer>, sequence<OuterIs...>,
                sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                Args &&...args, const int scratch_level,
                const std::size_t scratch_size_in_bytes) {
    const std::size_t size =
        ((bound_arr[OuterIs].e - bound_arr[OuterIs].s + 1) * ...);
    kokkos_dispatch(
        Tag(), name,
        team_policy(exec_space, size, Kokkos::AUTO)
            .set_scratch_size(scratch_level,
                              Kokkos::PerTeam(scratch_size_in_bytes)),
        KOKKOS_LAMBDA(team_mbr_t team_member, ExtraFuncArgs... fargs) {
          const auto idxer =
              make_indexer(Kokkos::Array<IndexRange, sizeof...(OuterIs)>{
                  bound_arr[OuterIs]...});
          const auto idx_arr = idxer.get_idx_array(team_member.league_rank());
          function(team_member, idx_arr[OuterIs]...,
                   std::forward<ExtraFuncArgs>(fargs)...);
        },
        std::forward<Args>(args)...);
  }

  // Collapse inner Nvector + Nthread loops to thread/vector range policies and
  // remaining outer loops to a team_policy.
  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void
  dispatch_impl(LoopPatternTag<PT::collapse>, sequence<OuterIs...>,
                sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                Args &&...args, const int scratch_level,
                const std::size_t scratch_size_in_bytes) {
    const auto idxer = make_indexer(
        Kokkos::Array<IndexRange, sizeof...(OuterIs)>{bound_arr[OuterIs]...});
    using HierarchicalPar = typename dispatch_type::HierarchicalPar;
    constexpr std::size_t Nvector = HierarchicalPar::Nvector;
    constexpr std::size_t Nthread = HierarchicalPar::Nthread;
    constexpr std::size_t Nouter = Rank - Nvector - Nthread;
    kokkos_dispatch(
        Tag(), name,
        team_policy(exec_space, idxer.size(), Kokkos::AUTO)
            .set_scratch_size(scratch_level,
                              Kokkos::PerTeam(scratch_size_in_bytes)),

        make_collapse<Rank, Nouter, Nthread, Nvector, ExtraFuncArgs...>(
            idxer, bound_arr, function),
        std::forward<Args>(args)...);
  }
};

template <typename Tag, typename Pattern, typename... AllArgs>
inline void par_dispatch(Pattern, std::string name, DevExecSpace exec_space,
                         AllArgs &&...args) {
  using dispatchsig = DispatchSignature<TypeList<AllArgs...>>;
  constexpr std::size_t Rank = dispatchsig::Rank;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;
  using Args = typename dispatchsig::Args;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;

  if constexpr (Rank > 1 &&
                std::is_same_v<dispatch_impl::ParallelScanDispatch, Tag>) {
    static_assert(always_false<Tag>, "par_scan only for 1D loops");
  }
  ParDispatchImpl<Tag, Pattern, Function, LoopBounds, Args, ExtraFuncArgs>()
      .dispatch(name, exec_space, std::forward<AllArgs>(args)...);
}

template <typename Tag, typename... Args>
inline void par_dispatch(const std::string &name, Args &&...args) {
  par_dispatch<Tag>(DEFAULT_LOOP_PATTERN, name, DevExecSpace(),
                    std::forward<Args>(args)...);
}

template <std::size_t Rank, std::size_t... OuterIs, typename Function>
KOKKOS_INLINE_FUNCTION void
sequential_for(std::index_sequence<OuterIs...>, Function function,
               Kokkos::Array<IndexRange, Rank> bounds) {
  const auto idxer = make_indexer(
      std::pair<int, int>(bounds[OuterIs].s, bounds[OuterIs].e)...);
  for (int idx = 0; idx < idxer.size(); idx++) {
    const auto indices = idxer.get_idx_array(idx);
    function(indices[OuterIs]...);
  }
}

template <class, class>
struct SeqForImpl {};

template <class Function, class... Bounds>
struct SeqForImpl<Function, TypeList<Bounds...>> {
  KOKKOS_INLINE_FUNCTION void execute(Bounds &&...bounds, Function function) {
    using bound_translator = LoopBoundTranslator<Bounds...>;
    constexpr std::size_t Rank = bound_translator::Rank;
    const auto bound_arr =
        bound_translator().GetIndexRanges(std::forward<Bounds>(bounds)...);
    sequential_for(std::make_index_sequence<Rank>(), function, bound_arr);
  }
};

template <class... Args>
KOKKOS_INLINE_FUNCTION void seq_for(Args &&...args) {
  using dispatchsig = DispatchSignature<TypeList<Args...>>;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;

  SeqForImpl<Function, LoopBounds>().execute(std::forward<Args>(args)...);
}

template <class... Args>
inline void par_for(Args &&...args) {
  par_dispatch<dispatch_impl::ParallelForDispatch>(std::forward<Args>(args)...);
}

template <class... Args>
inline void par_reduce(Args &&...args) {
  par_dispatch<dispatch_impl::ParallelReduceDispatch>(
      std::forward<Args>(args)...);
}

template <class... Args>
inline void par_scan(Args &&...args) {
  par_dispatch<dispatch_impl::ParallelScanDispatch>(
      std::forward<Args>(args)...);
}

template <typename Pattern, typename... AllArgs>
  requires std::is_same_v<Pattern, OuterLoopPatternTeams>
inline void par_for_outer(Pattern, const std::string &name,
                          DevExecSpace exec_space,
                          std::size_t scratch_size_in_bytes,
                          const int scratch_level, AllArgs &&...args) {
  using dispatchsig = DispatchSignature<TypeList<AllArgs...>>;
  static constexpr std::size_t Rank = dispatchsig::Rank;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;
  using Args = typename dispatchsig::Args;
  using Tag = dispatch_impl::ParallelForDispatch;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;

  ParDispatchImpl<Tag, Pattern, Function, LoopBounds, Args, ExtraFuncArgs>()
      .dispatch(name, exec_space, std::forward<AllArgs>(args)..., scratch_level,
                scratch_size_in_bytes);
}

template <typename... Args>
inline void par_for_outer(const std::string &name, Args &&...args) {
  par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, name, DevExecSpace(),
                std::forward<Args>(args)...);
}

template <typename Pattern, typename... AllArgs>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(Pattern, team_mbr_t team_member,
                                               AllArgs &&...args) {
  par_disp_inner(Pattern(), team_member, std::forward<AllArgs>(args)...);
}

template <typename... Args>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(team_mbr_t team_member,
                                               Args &&...args) {
  par_for_inner(DEFAULT_INNER_LOOP_PATTERN, team_member,
                std::forward<Args>(args)...);
}

// Inner reduction loops
template <typename Function, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
par_reduce_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int kl,
                 const int ku, const int jl, const int ju, const int il,
                 const int iu, const Function &function, T reduction) {
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, NkNjNi),
      [&](const int &idx, typename T::value_type &lreduce) {
        int k = idx / NjNi;
        int j = (idx - k * NjNi) / Ni;
        int i = idx - k * NjNi - j * Ni;
        k += kl;
        j += jl;
        i += il;
        function(k, j, i, lreduce);
      },
      reduction);
}

template <typename Function, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
par_reduce_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int jl,
                 const int ju, const int il, const int iu,
                 const Function &function, T reduction) {
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, NjNi),
      [&](const int &idx, typename T::value_type &lreduce) {
        int j = idx / Ni;
        int i = idx - j * Ni;
        j += jl;
        i += il;
        function(j, i, lreduce);
      },
      reduction);
}

template <typename Function, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
par_reduce_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int il,
                 const int iu, const Function &function, T reduction) {
  const int Ni = iu - il + 1;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, Ni),
      [&](const int &idx, typename T::value_type &lreduce) {
        int i = idx;
        i += il;
        function(i, lreduce);
      },
      reduction);
}

namespace custom_reductions { // namespace helps with name resolution in
                              // reduction identity
template <class ScalarType, int N>
struct array_type {
  ScalarType data[N];

  KOKKOS_INLINE_FUNCTION // Default constructor - Initialize to 0's
  array_type() {
    for (int i = 0; i < N; i++) {
      data[i] = 0;
    }
  }
  KOKKOS_INLINE_FUNCTION // Copy Constructor
  array_type(const array_type &rhs) {
    for (int i = 0; i < N; i++) {
      data[i] = rhs.data[i];
    }
  }
  KOKKOS_INLINE_FUNCTION // add operator
      array_type &operator+=(const array_type &src) {
    for (int i = 0; i < N; i++) {
      data[i] += src.data[i];
    }
    return *this;
  }
};
using ValueType = array_type<double, 4>;

struct ArgMax {
  int index;
  double value;

  KOKKOS_INLINE_FUNCTION
  ArgMax() : index(-1), value(-Kokkos::reduction_identity<double>::min()) {}

  KOKKOS_INLINE_FUNCTION
  ArgMax(int idx, double val) : index(idx), value(val) {}

  KOKKOS_INLINE_FUNCTION
  auto operator=(const ArgMax &other) -> ArgMax & = default;

  KOKKOS_INLINE_FUNCTION
  auto operator>(const ArgMax &rhs) const -> bool {
    return this->value > rhs.value;
  }
};
} // namespace custom_reductions

} // namespace athelas

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<athelas::custom_reductions::ValueType> {
  KOKKOS_FORCEINLINE_FUNCTION static athelas::custom_reductions::ValueType
  sum() {
    return athelas::custom_reductions::ValueType();
  }
};
template <>
struct reduction_identity<athelas::custom_reductions::ArgMax> {
  KOKKOS_FORCEINLINE_FUNCTION
  static athelas::custom_reductions::ArgMax max() {
    return athelas::custom_reductions::ArgMax(
        -1, -Kokkos::reduction_identity<double>::max());
  }
};
} // namespace Kokkos
