@fieldwise_init
struct SparseDoseEntry(Copyable, Movable):
    var voxel: Int
    var spot: Int
    var value: Float64


@fieldwise_init
struct SparseDoseMatrix(Copyable, Movable):
    var voxel_count: Int
    var spot_count: Int
    var entries: List[SparseDoseEntry]
