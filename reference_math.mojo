"""Host math calls that must match the TRiP CPU reference."""

from std.ffi import external_call


@always_inline("nodebug")
def reference_exp(value: Float64) -> Float64:
    return external_call["exp", Float64](value)
