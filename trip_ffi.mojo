from std.ffi import OwnedDLHandle, RTLD


struct TripFFI(Movable):
    var lib: OwnedDLHandle
    var rst_pnt_dose_distance_dr2: def(
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
    ) thin abi("C") -> Float64
    var ddd2_interp_col_fn: def(
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Float64,
        Int32,
    ) thin abi("C") -> Float64
    var dedx_eval_fn: def(
        Int32,
        Int32,
        Float64,
    ) thin abi("C") -> Float64

    def __init__(out self, path: String = "build/ffi/libtripffi.so") raises:
        self.lib = OwnedDLHandle(path, RTLD.LAZY | RTLD.LOCAL)
        self.rst_pnt_dose_distance_dr2 = self.lib.get_function[
            def(
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
            ) thin abi("C") -> Float64
        ]("tripffi_rst_pnt_dose_distance_dr2")
        self.ddd2_interp_col_fn = self.lib.get_function[
            def(
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Int32,
            ) thin abi("C") -> Float64
        ]("tripffi_ddd2_interp_col")
        self.dedx_eval_fn = self.lib.get_function[
            def(
                Int32,
                Int32,
                Float64,
            ) thin abi("C") -> Float64
        ]("tripffi_dedx_eval")

    def dose_distance_dr2(
        self,
        point_x: Float64,
        point_y: Float64,
        f2_max: Float64,
        voxel_x: Float64,
        voxel_y: Float64,
        div_x: Float64,
        div_y: Float64,
    ) -> Float64:
        return self.rst_pnt_dose_distance_dr2(
            point_x,
            point_y,
            f2_max,
            voxel_x,
            voxel_y,
            div_x,
            div_y,
        )

    def ddd2_interp_col(
        self,
        z0: Float64,
        d0: Float64,
        fwhm10: Float64,
        mix0: Float64,
        fwhm20: Float64,
        z1: Float64,
        d1: Float64,
        fwhm11: Float64,
        mix1: Float64,
        fwhm21: Float64,
        depth_cm: Float64,
        column: Int32,
    ) -> Float64:
        return self.ddd2_interp_col_fn(
            z0,
            d0,
            fwhm10,
            mix0,
            fwhm20,
            z1,
            d1,
            fwhm11,
            mix1,
            fwhm21,
            depth_cm,
            column,
        )

    def dedx_eval(self, z: Int32, a: Int32, energy: Float64) -> Float64:
        return self.dedx_eval_fn(z, a, energy)
