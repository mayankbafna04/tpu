import numpy as np
import struct
import os

MAT_SIZE = 16

input_data = np.zeros((MAT_SIZE, MAT_SIZE), dtype=np.int8)
for i in range(MAT_SIZE):
    input_data[i, :] = i + 1

weight_data = np.zeros((MAT_SIZE, MAT_SIZE), dtype=np.int8)
for i in range(MAT_SIZE):
    weight_data[i, i] = -1

HOST_MEM_SIZE = 4 * 1024 * 1024
host_memory_buffer = bytearray(HOST_MEM_SIZE)

ADDR_INPUT = 1000
ADDR_WEIGHTS = 2000
ADDR_RESULT = 3000

host_memory_buffer[ADDR_INPUT : ADDR_INPUT + 256] = input_data.tobytes()
host_memory_buffer[ADDR_WEIGHTS : ADDR_WEIGHTS + 256] = weight_data.tobytes()

OP_RHM = 0x01
OP_WHM = 0x02
OP_RW  = 0x03
OP_MMC = 0x04
OP_ACT = 0x05
OP_HLT = 0xFF

instr_fmt = struct.Struct('<B 3x I I I') # 16 bytes

program = [
    (OP_RHM, 0, ADDR_INPUT, 256),
    (OP_RW,  0, ADDR_WEIGHTS, 256),
    (OP_MMC, 0, 0, 256), 
    (OP_ACT, 0, 0, 256), 
    (OP_WHM, 0, ADDR_RESULT, 1024), 
    (OP_HLT, 0, 0, 0)
]

with open("program.bin", "wb") as f:
    for instr in program:
        f.write(instr_fmt.pack(*instr))

with open("memory.bin", "wb") as f:
    f.write(host_memory_buffer)

# Validation check
py_result = input_data.astype(np.int32) @ weight_data.astype(np.int32)
py_result = np.maximum(0, py_result) # ReLU
print(f"Python validation: A[0,0]({input_data[0,0]}) * W[0,0]({weight_data[0,0]}) -> ReLU -> {py_result[0,0]}")
