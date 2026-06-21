#include <string>
#include <utility>

#include "bc/boundary_conditions.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "utils/error.hpp"

namespace athelas::bc {

namespace {
void fill_ghost_zone_range(AthelasArray3D<double> U, const IndexRange &vars,
                           const bool periodic) {
  const int num_nodes = static_cast<int>(U.extent(1));
  const int nx = static_cast<int>(U.extent(0)) - 2;
  const int inner_source = periodic ? nx : 1;
  const int outer_source = periodic ? 1 : nx;
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Fill ghosts", DevExecSpace(), vars.s, vars.e,
      KOKKOS_LAMBDA(const int v) {
        for (int k = 0; k < num_nodes; k++) {
          U(0, k, v) = U(inner_source, k, v); // inner ghost
          U(nx + 1, k, v) = U(outer_source, k, v); // outer ghost
        }
      });
}

void ghost_fill_array(AthelasArray3D<double> U, BoundaryConditions *bcs,
                      const bool radiation_enabled,
                      const bool composition_enabled) {
  static const int nvars = static_cast<int>(U.extent(2));
  static const bool fluid_periodic = bcs->fluid_bc[0].type == BcType::Periodic;

  fill_ghost_zone_range(U, IndexRange(std::make_pair(0, 2)), fluid_periodic);

  int composition_start = 3;
  if (radiation_enabled) {
    const bool rad_periodic = bcs->rad_bc[0].type == BcType::Periodic;
    fill_ghost_zone_range(U, IndexRange(std::make_pair(3, 4)), rad_periodic);
    composition_start = 5;
  }

  if (composition_enabled && nvars > composition_start) {
    fill_ghost_zone_range(
        U, IndexRange(std::make_pair(composition_start, nvars - 1)),
        fluid_periodic);
  }
}
} // namespace

void ghost_fill(const MeshState &mesh_state, BoundaryConditions *bcs) {
  auto evolved = mesh_state(0).get_field("evolved");
  ghost_fill_array(evolved, bcs, mesh_state.enabled("radiation"),
                   mesh_state.enabled("composition"));
}

void ghost_fill(const StageData &stage_data, BoundaryConditions *bcs) {
  auto evolved = stage_data.get_field("evolved");
  ghost_fill(evolved, stage_data, bcs);
}

void ghost_fill_derived(AthelasArray3D<double> derived,
                        BoundaryConditions *bcs) {
  const int nderived = static_cast<int>(derived.extent(2));
  const bool fluid_periodic = bcs->fluid_bc[0].type == BcType::Periodic;
  // All derived quantities follow the (fluid) domain topology: periodic wraps,
  // otherwise the ghost is a copy of the adjacent interior cell.
  fill_ghost_zone_range(derived, IndexRange(std::make_pair(0, nderived - 1)),
                        fluid_periodic);
}

void ghost_fill(AthelasArray3D<double> U, const StageData &stage_data,
                BoundaryConditions *bcs) {
  ghost_fill_array(U, bcs, stage_data.enabled("radiation"),
                   stage_data.enabled("composition"));
}

auto parse_bc_type(const std::string &name) -> BcType {
  if (name == "outflow") {
    return BcType::Outflow;
  }
  if (name == "interior") {
    return BcType::InteriorFlux;
  }
  if (name == "free_streaming") {
    return BcType::FreeStreaming;
  }
  if (name == "reflecting") {
    return BcType::Reflecting;
  }
  if (name == "periodic") {
    return BcType::Periodic;
  }
  if (name == "marshak") {
    return BcType::Marshak;
  }
  if (name == "surface") {
    return BcType::Surface;
  }
  throw_athelas_error(
      " ! Initialization Error: Bad boundary condition choice. Choose: \n"
      " - outflow \n"
      " - interior \n"
      " - free_streaming \n"
      " - reflecting \n"
      " - periodic \n"
      " - marshak \n"
      " - surface");
  return BcType::Null;
}

void validate_fluid_bc(const BcType type) {
  switch (type) {
  case BcType::Outflow:
  case BcType::Reflecting:
  case BcType::Periodic:
  case BcType::Surface:
    return;
  case BcType::Marshak:
    throw_athelas_error("Marshak is only valid for radiation boundaries.");
  case BcType::InteriorFlux:
  case BcType::FreeStreaming:
    throw_athelas_error("Use 'outflow' for fluid boundaries; 'interior' and "
                        "'free_streaming' are radiation-only.");
  case BcType::Null:
    break;
  }
  throw_athelas_error("Null BC is not valid.");
}

void validate_radiation_bc(const Boundary side, const BcType type) {
  switch (type) {
  case BcType::Reflecting:
  case BcType::Periodic:
  case BcType::InteriorFlux:
  case BcType::FreeStreaming:
    return;
  case BcType::Marshak:
    if (side == Boundary::Exterior) {
      throw_athelas_error(
          "Outer Marshak radiation boundaries are not implemented.");
    }
    return;
  case BcType::Outflow:
    throw_athelas_error(
        "Use 'interior' or 'free_streaming' for radiation boundaries; "
        "'outflow' is fluid-only.");
  case BcType::Surface:
    throw_athelas_error("Surface is only valid for fluid boundaries.");
  case BcType::Null:
    break;
  }
  throw_athelas_error("Null BC is not valid.");
}

auto make_boundary_conditions(const ProblemIn *pin) -> BoundaryConditions {
  BoundaryConditions my_bc;
  const auto do_rad = pin->param()->get<bool>("physics.radiation.enabled");
  const auto fluid_bc_i = pin->param()->get<std::string>("fluid.bc.i");
  const auto fluid_bc_o = pin->param()->get<std::string>("fluid.bc.o");
  const auto fluid_i_surface_pressure =
      pin->param()->get<double>("fluid.bc.i.surface_pressure", 0.0);
  const auto fluid_o_surface_pressure =
      pin->param()->get<double>("fluid.bc.o.surface_pressure", 0.0);
  const auto rad_bc_i =
      pin->param()->get<std::string>("radiation.bc.i", "interior");
  const auto rad_bc_o =
      pin->param()->get<std::string>("radiation.bc.o", "interior");
  const auto rad_i_marshak_incoming_energy =
      pin->param()->get<double>("radiation.bc.i.marshak_incoming_energy", 0.0);
  const auto rad_o_marshak_incoming_energy =
      pin->param()->get<double>("radiation.bc.o.marshak_incoming_energy", 0.0);

  // --- Fluid BCs ---
  BcType f_inner = parse_bc_type(fluid_bc_i);
  BcType f_outer = parse_bc_type(fluid_bc_o);

  if (f_inner == BcType::Surface) {
    my_bc.fluid_bc[0] = make_surface_bc(fluid_i_surface_pressure);
  } else {
    my_bc.fluid_bc[0] = make_bc(f_inner);
  }

  if (f_outer == BcType::Surface) {
    my_bc.fluid_bc[1] = make_surface_bc(fluid_o_surface_pressure);
  } else {
    my_bc.fluid_bc[1] = make_bc(f_outer);
  }

  // --- Radiation BCs ---
  if (do_rad) {
    if (rad_bc_i == "" || rad_bc_o == "") {
      throw_athelas_error(" ! Radiation enabled but rad_bc_i/o is not set.");
    }

    my_bc.do_rad = true;

    BcType r_inner = parse_bc_type(rad_bc_i);
    BcType r_outer = parse_bc_type(rad_bc_o);

    if (r_inner == BcType::Marshak) {
      my_bc.rad_bc[0] = make_marshak_bc(rad_i_marshak_incoming_energy);
    } else {
      my_bc.rad_bc[0] = make_bc(r_inner);
    }

    if (r_outer == BcType::Marshak) {
      my_bc.rad_bc[1] = make_marshak_bc(rad_o_marshak_incoming_energy);
    } else {
      my_bc.rad_bc[1] = make_bc(r_outer);
    }
  }

  return my_bc;
}
} // namespace athelas::bc
