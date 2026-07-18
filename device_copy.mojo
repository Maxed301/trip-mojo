"""Small host-to-device copy helpers shared by accelerator backends."""

from std.gpu.host import DeviceBuffer, DeviceContext
from std.memory import UnsafePointer


def copy_bytes_to_device[
    origin: Origin, //
](
    context: DeviceContext,
    source: UnsafePointer[UInt8, origin],
    byte_count: Int,
) raises -> DeviceBuffer[DType.uint8]:
    var allocation_count = byte_count
    if allocation_count == 0:
        allocation_count = 1
    var host = context.enqueue_create_host_buffer[DType.uint8](allocation_count)
    for index in range(byte_count):
        host[index] = source[index]
    var device = context.enqueue_create_buffer[DType.uint8](allocation_count)
    context.enqueue_copy(device, host)
    return device^
