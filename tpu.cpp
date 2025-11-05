#include "tpu.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>

int LATENCY_HOST_MEM_READ = 100;
int LATENCY_HOST_MEM_WRITE = 100;

TPU::TPU(size_t host_memory_size_mb) 
    : controller_state(ControllerState::FETCH), instruction_pointer(0),
      host_mem_state(HostMemState::IDLE), host_mem_cycles_remaining(0) {
    host_memory.resize(host_memory_size_mb * 1024 * 1024, 0); 
    std::cout << "--- Booting C++ TPU Simulator ---" << std::endl;
}

void TPU::load_program(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { std::cerr << "ERROR: Bad program file: " << filepath << std::endl; return; }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size % sizeof(Instruction) != 0) {
        std::cerr << "ERROR: Program file size is wrong!" << std::endl;
        return;
    }
    program.resize(size / sizeof(Instruction));
    if (!file.read(reinterpret_cast<char*>(program.data()), size)) {
        std::cerr << "ERROR: Failed to read program file." << std::endl;
    }
}

void TPU::load_host_memory(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { std::cerr << "ERROR: Bad memory file: " << filepath << std::endl; return; }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size > host_memory.size()) { std::cerr << "ERROR: Memory file too big!" << std::endl; return; }
    if (!file.read(reinterpret_cast<char*>(host_memory.data()), size)) {
        std::cerr << "ERROR: Failed to read memory file." << std::endl;
    }
}

void TPU::tick_host_memory() {
    if (host_mem_state == HostMemState::BUSY) {
        host_mem_cycles_remaining--;
        if (host_mem_cycles_remaining <= 0) {
            host_mem_state = HostMemState::IDLE;
        }
    }
}

bool TPU::host_read_request(uint32_t addr, uint32_t length) {
    if (host_mem_state == HostMemState::BUSY) return false;
    host_mem_state = HostMemState::BUSY;
    host_mem_cycles_remaining = LATENCY_HOST_MEM_READ;
    data_buffer_a.clear();
    data_buffer_a.resize(length, 0);
    for (uint32_t i = 0; i < length; ++i) {
        if (addr + i < host_memory.size()) data_buffer_a[i] = host_memory[addr + i];
    }
    return true;
}

bool TPU::host_write_request(uint32_t addr, const std::vector<uint8_t>& data) {
    if (host_mem_state == HostMemState::BUSY) return false;
    host_mem_state = HostMemState::BUSY;
    host_mem_cycles_remaining = LATENCY_HOST_MEM_WRITE;
    for (size_t i = 0; i < data.size(); ++i) {
        if (addr + i < host_memory.size()) host_memory[addr + i] = data[i];
    }
    return true;
}

void TPU::tick() {
    stats.total_cycles++;
    if (unified_buffer.get_state() == CompState::BUSY) stats.ub_busy_cycles++;
    if (systolic_array.get_state() == CompState::BUSY) stats.mxu_busy_cycles++;
    if (accumulator.get_state() == CompState::BUSY) stats.acc_busy_cycles++;
    if (host_mem_state == HostMemState::BUSY) stats.host_mem_busy_cycles++;

    unified_buffer.tick();
    weight_fifo.tick();
    systolic_array.tick();
    accumulator.tick();
    tick_host_memory();

    switch (controller_state) {
        case ControllerState::FETCH:   tick_fetch();   break;
        case ControllerState::DECODE:  tick_decode();  break;
        case ControllerState::HALTED:  break;
        default:                       tick_execute(); break;
    }
}

void TPU::tick_fetch() {
    if (instruction_pointer >= program.size()) {
        controller_state = ControllerState::HALTED;
        return;
    }
    current_instruction = program[instruction_pointer];
    instruction_pointer++;
    stats.instruction_count++;
    controller_state = ControllerState::DECODE;
}

void TPU::tick_decode() {
    switch (current_instruction.opcode) {
        case OpCode::RHM: controller_state = ControllerState::EXECUTE_RHM_READ_HOST; break;
        case OpCode::WHM: controller_state = ControllerState::EXECUTE_WHM_READ_ACC;  break;
        case OpCode::RW:  controller_state = ControllerState::EXECUTE_RW_READ_HOST;  break;
        case OpCode::MMC: 
                    controller_state = ControllerState::EXECUTE_MMC_READ_UB;
                    stats.mmc_count++;
                    break;        
        case OpCode::ACT: controller_state = ControllerState::EXECUTE_ACT_RUN;       break;
        case OpCode::HLT: 
            std::cout << "CYCLE " << stats.total_cycles << ": DECODE -> HLT" << std::endl;
            controller_state = ControllerState::HALTED;
            break;
        default:
            std::cout << "CYCLE " << stats.total_cycles << ": ERROR: Unknown opcode" << std::endl;
            controller_state = ControllerState::HALTED;
            break;
    }
}

void TPU::tick_execute() {
    switch (controller_state) {
        case ControllerState::EXECUTE_RHM_READ_HOST:
            if (host_mem_state == HostMemState::IDLE) {
                host_read_request(current_instruction.host_addr, current_instruction.length);
                controller_state = ControllerState::EXECUTE_RHM_WRITE_UB;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_RHM_WRITE_UB:
            if (host_mem_state == HostMemState::IDLE && unified_buffer.get_state() == CompState::IDLE) {
                unified_buffer.write_request(current_instruction.data_addr, data_buffer_a);
                controller_state = ControllerState::FETCH;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_RW_READ_HOST:
            if (host_mem_state == HostMemState::IDLE) {
                host_read_request(current_instruction.host_addr, current_instruction.length);
                controller_state = ControllerState::FETCH; 
                weight_fifo.load(data_buffer_a); 
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_MMC_READ_UB:
            if (unified_buffer.get_state() == CompState::IDLE) {
                unified_buffer.read_request(current_instruction.data_addr, current_instruction.length);
                controller_state = ControllerState::EXECUTE_MMC_READ_FIFO;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_MMC_READ_FIFO:
            if (unified_buffer.get_state() == CompState::IDLE) { 
                data_buffer_a = unified_buffer.get_read_result();
                data_buffer_b = weight_fifo.read();
                controller_state = ControllerState::EXECUTE_MMC_EXECUTE;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_MMC_EXECUTE:
            if (systolic_array.get_state() == CompState::IDLE) {
                systolic_array.execute_request(data_buffer_a, data_buffer_b);
                controller_state = ControllerState::EXECUTE_MMC_WRITE_ACC;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_MMC_WRITE_ACC:
            if (systolic_array.get_state() == CompState::IDLE && accumulator.get_state() == CompState::IDLE) {
                data_buffer_a = systolic_array.get_result();
                accumulator.write_request(current_instruction.host_addr, data_buffer_a);
                controller_state = ControllerState::FETCH;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_ACT_RUN:
            if (accumulator.get_state() == CompState::IDLE) {
                accumulator.activate_request(current_instruction.data_addr, current_instruction.length);
                controller_state = ControllerState::FETCH;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_WHM_READ_ACC:
            if (accumulator.get_state() == CompState::IDLE) {
                accumulator.read_request(current_instruction.data_addr, current_instruction.length);
                controller_state = ControllerState::EXECUTE_WHM_WRITE_HOST;
            } else { stats.stall_cycles++; }
            break;
        case ControllerState::EXECUTE_WHM_WRITE_HOST:
            if (accumulator.get_state() == CompState::IDLE && host_mem_state == HostMemState::IDLE) {
                data_buffer_a = accumulator.get_read_result();
                host_write_request(current_instruction.host_addr, data_buffer_a);
                if (data_buffer_a.size() >= 4) {
                    int32_t first_result;
                    std::memcpy(&first_result, data_buffer_a.data(), sizeof(int32_t));
                    std::cout << "CYCLE " << stats.total_cycles << ": WHM Issued. First 32-bit result: " << first_result << std::endl;
                }
                controller_state = ControllerState::FETCH;
            } else { stats.stall_cycles++; }
            break;
        default:
            controller_state = ControllerState::HALTED;
            break;
    }
}

void TPU::print_performance_report() {
    std::cout << "\n--- PERFORMANCE REPORT ---" << std::endl;
    if (stats.total_cycles == 0 || stats.instruction_count == 0) {
        std::cout << "No operations performed." << std::endl;
        return;
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Core Metrics:" << std::endl;
    std::cout << "  Total Cycles:       " << stats.total_cycles << std::endl;
    std::cout << "  Instructions Exec:  " << stats.instruction_count << std::endl;
    std::cout << "  Cycles Per Instr (CPI): " << (double)stats.total_cycles / stats.instruction_count << std::endl;
    
    double stall_percent = (double)stats.stall_cycles / stats.total_cycles * 100.0;
    std::cout << "\nStall Analysis:" << std::endl;
    std::cout << "  Controller Stall Cycles: " << stats.stall_cycles << " (" << stall_percent << " % of total)" << std::endl;

    double host_util = (double)stats.host_mem_busy_cycles / stats.total_cycles * 100.0;
    double ub_util = (double)stats.ub_busy_cycles / stats.total_cycles * 100.0;
    double acc_util = (double)stats.acc_busy_cycles / stats.total_cycles * 100.0;
    double mxu_util = (double)stats.mxu_busy_cycles / stats.total_cycles * 100.0;
    std::cout << "\nComponent Utilization:" << std::endl;
    std::cout << "  Host Memory Bus:  " << stats.host_mem_busy_cycles << " cycles (" << host_util << " %)" << std::endl;
    std::cout << "  Unified Buffer (UB): " << stats.ub_busy_cycles << " cycles (" << ub_util << " %)" << std::endl;
    std::cout << "  Accumulator (ACC): " << stats.acc_busy_cycles << " cycles (" << acc_util << " %)" << std::endl;
    std::cout << "  Matrix Unit (MXU): " << stats.mxu_busy_cycles << " cycles (" << mxu_util << " %)" << std::endl;

    const double OPS_PER_MMC = 16.0 * 16.0 * 16.0 * 2.0;
    const double CLOCK_SPEED_MHZ = 500.0; 
    double total_ops = (double)stats.mmc_count * OPS_PER_MMC;
    double total_time_sec = (double)stats.total_cycles / (CLOCK_SPEED_MHZ * 1e6);
    double gops = (total_ops / total_time_sec) / 1e9;

    std::cout << "\nPerformance (Assuming " << CLOCK_SPEED_MHZ << " MHz Clock):" << std::endl;
    std::cout << "  Total Operations (MACs): " << total_ops / 2.0 << std::endl;
    std::cout << "  Total Time:          " << total_time_sec * 1e6 << " us" << std::endl;
    std::cout << "  Effective GOPS:      " << gops << std::endl;
    std::cout << "--- END OF REPORT ---" << std::endl;
}
