#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <queue>

using MemoryModel = std::map<uint32_t, uint8_t>;

enum class CompState {
    IDLE,
    BUSY
};

class UnifiedBuffer {
private:
    MemoryModel memory;
    size_t size_bytes;
    CompState state;
    int cycles_remaining;
    std::vector<uint8_t> write_data_buffer;
    std::vector<uint8_t> read_result_buffer;
    uint32_t op_addr;
    uint32_t op_length;

    void write_internal();
    void read_internal();

public:
    UnifiedBuffer(size_t size_kb = 256);
    void tick(); 
    bool read_request(uint32_t addr, uint32_t length);
    bool write_request(uint32_t addr, const std::vector<uint8_t>& data);
    std::vector<uint8_t> get_read_result();
    std::vector<uint8_t> read(uint32_t addr, uint32_t length);
    void write(uint32_t addr, const std::vector<uint8_t>& data);
    CompState get_state() { return state; }
};

class WeightFIFO {
private:
    std::queue<std::vector<uint8_t>> fifo;
    CompState state;
public:
    WeightFIFO();
    void tick();
    void load(const std::vector<uint8_t>& weights);
    std::vector<uint8_t> read(); 
    CompState get_state() { return state; }
};

class SystolicArray {
private:
    int size;
    CompState state;
    int cycles_remaining;
    std::vector<uint8_t> input_buffer;
    std::vector<uint8_t> weight_buffer;
    std::vector<uint8_t> result_buffer; 

    void execute_internal();
public:
    SystolicArray(int size = 16);
    void tick();
    bool execute_request(const std::vector<uint8_t>& inputs, const std::vector<uint8_t>& weights);
    std::vector<uint8_t> get_result(); 
    std::vector<uint8_t> execute(const std::vector<uint8_t>& inputs, const std::vector<uint8_t>& weights);
    CompState get_state() { return state; }
};

class Accumulator {
private:
    MemoryModel memory;
    size_t size;
    CompState state;
    int cycles_remaining;
    enum class AccOp { WRITE, READ, ACTIVATE };
    AccOp pending_op;
    std::vector<uint8_t> write_data_buffer;
    std::vector<uint8_t> read_result_buffer;
    uint32_t op_addr;
    uint32_t op_length_or_elements;

    void write_internal();
    void read_internal();
    void activate_internal();

public:
    Accumulator(size_t entries = 4096);
    void tick();
    bool write_request(uint32_t addr, const std::vector<uint8_t>& data);
    bool read_request(uint32_t addr, uint32_t length);
    bool activate_request(uint32_t addr, uint32_t num_elements);
    std::vector<uint8_t> get_read_result();
    std::vector<uint8_t> read(uint32_t addr, uint32_t length);
    void write(uint32_t addr, const std::vector<uint8_t>& data);
    void activate(uint32_t addr, uint32_t num_elements);
    CompState get_state() { return state; }
};
