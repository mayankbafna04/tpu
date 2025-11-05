#pragma once
#include "isa.h"
#include "tpu_components.h"
#include <vector>
#include <string>

enum class ControllerState {
    FETCH, DECODE,
    EXECUTE_RHM_READ_HOST, EXECUTE_RHM_WRITE_UB,
    EXECUTE_RW_READ_HOST,
    EXECUTE_MMC_READ_UB, EXECUTE_MMC_READ_FIFO, EXECUTE_MMC_EXECUTE, EXECUTE_MMC_WRITE_ACC,
    EXECUTE_ACT_RUN,
    EXECUTE_WHM_READ_ACC, EXECUTE_WHM_WRITE_HOST,
    HALTED
};

enum class HostMemState { IDLE, BUSY };

class TPU {
public:
    struct PerformanceStats {
        uint64_t total_cycles;
        uint64_t instruction_count;
        uint64_t stall_cycles;
        uint64_t host_mem_busy_cycles;
        uint64_t ub_busy_cycles;
        uint64_t acc_busy_cycles;
        uint64_t mxu_busy_cycles;
        uint64_t mmc_count;
        
        PerformanceStats() : total_cycles(0), instruction_count(0), stall_cycles(0),
                             host_mem_busy_cycles(0), ub_busy_cycles(0), acc_busy_cycles(0),
                             mxu_busy_cycles(0), mmc_count(0) {}
    };

    TPU(size_t host_memory_size_mb = 4);
    void load_program(const std::string& filepath);
    void load_host_memory(const std::string& filepath);
    void tick();
    bool is_halted() { return controller_state == ControllerState::HALTED; }
    uint64_t get_cycle_count() { return stats.total_cycles; }
    void print_performance_report();

private:
    UnifiedBuffer unified_buffer;
    WeightFIFO weight_fifo;
    SystolicArray systolic_array;
    Accumulator accumulator;
    
    ControllerState controller_state;
    uint32_t instruction_pointer;
    std::vector<Instruction> program;
    Instruction current_instruction;
    
    std::vector<uint8_t> host_memory;
    HostMemState host_mem_state;
    int host_mem_cycles_remaining;
    
    std::vector<uint8_t> data_buffer_a;
    std::vector<uint8_t> data_buffer_b;

    PerformanceStats stats;

    void tick_fetch();
    void tick_decode();
    void tick_execute();
    
    bool host_read_request(uint32_t addr, uint32_t length);
    bool host_write_request(uint32_t addr, const std::vector<uint8_t>& data);
    void tick_host_memory();
};
