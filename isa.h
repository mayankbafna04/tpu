#pragma once
#include <cstdint>

enum class OpCode : uint8_t {
    RHM = 0x01,
    WHM = 0x02,
    RW  = 0x03,
    MMC = 0x04,
    ACT = 0x05,
    HLT = 0xFF
};

struct Instruction {
    OpCode opcode;
    uint32_t data_addr;
    uint32_t host_addr;
    uint32_t length;
};
