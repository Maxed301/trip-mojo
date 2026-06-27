from case_model import NativeCase, NativeField
from raster_grid import RasterGrid2D


@fieldwise_init
struct FieldRasterMarkCounts(Copyable, Movable):
    var total_raster_points: Int
    var target_inside_points: Int
    var robust_scenario_points: Int
    var extension_points: Int

    def optimization_point_count(self) -> Int:
        # TRiP's CT/FDCB optimizer counts every point flagged INSIDE in the
        # in-memory raster working copy. For robust P101 fields that set is the
        # union reported by MarkInsideUltimate: target-inside + robust-scenario
        # + extension points. It is intentionally not the written RST spot count.
        return self.target_inside_points + self.robust_scenario_points + self.extension_points


@fieldwise_init
struct OptimizationFieldSpec(Copyable, Movable):
    var field: NativeField
    var mark_counts: FieldRasterMarkCounts
    var written_rst_spots: Int

    def optimization_points(self) -> Int:
        return self.mark_counts.optimization_point_count()

    def omitted_from_written_rst(self) -> Int:
        return self.optimization_points() - self.written_rst_spots


@fieldwise_init
struct OptimizationSpot(Copyable, Movable):
    var field_index: Int
    var energy_index: Int
    var raster_point_index: Int
    var x: Float64
    var y: Float64
    var range_mm_h2o: Float64
    var energy_mev_u: Float64
    var focus_mm: Float64
    var particles: Float64
    var inside: Bool


@fieldwise_init
struct OptimizationFieldState(Copyable, Movable):
    var spec: OptimizationFieldSpec
    var grid: RasterGrid2D
    var spots: List[OptimizationSpot]

    def active_spots(self) -> Int:
        var total = 0
        for i in range(len(self.spots)):
            if self.spots[i].inside:
                total += 1
        return total

    def total_particles(self) -> Float64:
        var total = 0.0
        for i in range(len(self.spots)):
            if self.spots[i].inside:
                total += self.spots[i].particles
        return total


@fieldwise_init
struct OptimizationProblemSpec(Copyable, Movable):
    var native_case: NativeCase
    var fields: List[OptimizationFieldSpec]

    def optimization_points(self) -> Int:
        var total = 0
        for i in range(len(self.fields)):
            total += self.fields[i].optimization_points()
        return total

    def written_rst_spots(self) -> Int:
        var total = 0
        for i in range(len(self.fields)):
            total += self.fields[i].written_rst_spots
        return total

    def omitted_from_written_rst(self) -> Int:
        return self.optimization_points() - self.written_rst_spots()
