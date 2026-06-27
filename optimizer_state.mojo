from optimization_model import OptimizationFieldSpec, OptimizationFieldState, OptimizationSpot
from raster_grid import RasterGrid2D


def seed_uniform_optimization_field_state(
    field_index: Int,
    spec: OptimizationFieldSpec,
    grid: RasterGrid2D,
    energies: List[Float64],
    ranges_mm_h2o: List[Float64],
    initial_particles: Float64,
) raises -> OptimizationFieldState:
    if len(energies) != len(ranges_mm_h2o):
        raise Error("energy/range list length mismatch")
    var spots = List[OptimizationSpot]()
    spots.reserve(spec.optimization_points())
    var remaining = spec.optimization_points()
    var global_index = 0
    for ie in range(len(energies)):
        for ip in range(len(grid.points)):
            if remaining <= 0:
                break
            spots.append(OptimizationSpot(
                field_index,
                ie,
                grid.points[ip].index,
                grid.points[ip].x,
                grid.points[ip].y,
                ranges_mm_h2o[ie],
                energies[ie],
                0.0,
                initial_particles,
                True,
            ))
            remaining -= 1
            global_index += 1
    if remaining != 0:
        raise Error("raster grid/energy list cannot cover optimization point count")
    return OptimizationFieldState(spec.copy(), grid.copy(), spots^)


def apply_min_particles(mut state: OptimizationFieldState, min_particles: Float64):
    for i in range(len(state.spots)):
        if state.spots[i].inside and state.spots[i].particles < min_particles:
            state.spots[i].particles = 0.0
            state.spots[i].inside = False


def scale_particles(mut state: OptimizationFieldState, factor: Float64):
    for i in range(len(state.spots)):
        if state.spots[i].inside:
            state.spots[i].particles *= factor
