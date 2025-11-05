#include "tpu_components.h"
#include <stdexcept>
#include <cstring>

int LATENCY_SRAM_READ = 20;
int LATENCY_SRAM_WRITE = 20;
int LATENCY_ACC_READ = 5;
int LATENCY_ACC_WRITE = 5;
int LATENCY_ACTIVATE = 16;
int LATENCY_MXU = 32;

UnifiedBuffer::UnifiedBuffer(size_t size_kb) : state(CompState::IDLE), cycles_remaining(0) {
    this->size_bytes = size_kb * 1024;
}

void UnifiedBuffer::tick() {
    if (state == CompState::BUSY) {
        cycles_remaining--;
        if (cycles_remaining <= 0) {
            if (!write_data_buffer.empty()) {
                write_internal();
            } else {
                read_internal();
            }
            state = CompState::IDLE;
        }
    }
}

bool UnifiedBuffer::read_request(uint32_t addr, uint32_t length) {
    if (state == CompState::BUSY) return false;
    state = CompState::BUSY;
    cycles_remaining = LATENCY_SRAM_READ;
    this->op_addr = addr;
    this->op_length = length;
    this->write_data_buffer.clear();
    return true;
}

bool UnifiedBuffer::write_request(uint32_t addr, const std::vector<uint8_t>& data) {
    if (state == CompState::BUSY) return false;
    state = CompState::BUSY;
    cycles_remaining = LATENCY_SRAM_WRITE;
    this->op_addr = addr;
    this->write_data_buffer = data;
    return true;
}

std::vector<uint8_t> UnifiedBuffer::get_read_result() {
    return this->read_result_buffer;
}

void UnifiedBuffer::write_internal() {
    for (size_t i = 0; i < write_data_buffer.size(); ++i) { 
        this->memory[op_addr + i] = write_data_buffer[i]; 
    }
}

void UnifiedBuffer::read_internal() {
    read_result_buffer.clear();
    read_result_buffer.resize(op_length, 0);
    for (uint32_t i = 0; i < op_length; ++i) {
        auto it = this->memory.find(op_addr + i);
        if (it != this->memory.end()) { read_result_buffer[i] = it->second; }
    }
}

void UnifiedBuffer::write(uint32_t addr, const std::vector<uint8_t>& data) {
    for (size_t i = 0; i < data.size(); ++i) this->memory[addr + i] = data[i];
}

std::vector<uint8_t> UnifiedBuffer::read(uint32_t addr, uint32_t length) {
    std::vector<uint8_t> data_out(length, 0);
    for (uint32_t i = 0; i < length; ++i) {
        auto it = this->memory.find(addr + i);
        if (it != this->memory.end()) data_out[i] = it->second;
    }
    return data_out;
}

WeightFIFO::WeightFIFO() : state(CompState::IDLE) {}
void WeightFIFO::tick() {}
void WeightFIFO::load(const std::vector<uint8_t>& weights) {
    this->fifo.push(weights);
}
std::vector<uint8_t> WeightFIFO::read() {
    if (this->fifo.empty()) {
        return std::vector<uint8_t>();
    }
    std::vector<uint8_t> weights = this->fifo.front();
    this->fifo.pop();
    return weights;
}

SystolicArray::SystolicArray(int size) : size(size), state(CompState::IDLE), cycles_remaining(0) {}

void SystolicArray::tick() {
    if (state == CompState::BUSY) {
        cycles_remaining--;
        if (cycles_remaining <= 0) {
            execute_internal();
            state = CompState::IDLE;
        }
    }
}

bool SystolicArray::execute_request(const std::vector<uint8_t>& inputs, const std::vector<uint8_t>& weights) {
    if (state == CompState::BUSY) return false;
    state = CompState::BUSY;
    cycles_remaining = LATENCY_MXU;
    this->input_buffer = inputs;
    this->weight_buffer = weights;
    return true;
}

std::vector<uint8_t> SystolicArray::get_result() {
    return this->result_buffer;
}

void SystolicArray::execute_internal() {
    this->result_buffer = execute(this->input_buffer, this->weight_buffer);
}

std::vector<uint8_t> SystolicArray::execute(const std::vector<uint8_t>& inputs, const std::vector<uint8_t>& weights) {
    const int MAT_SIZE = 16;
    if (inputs.size() != 256 || weights.size() != 256) {
        return std::vector<uint8_t>();
    }
    std::vector<int32_t> results_32bit(MAT_SIZE * MAT_SIZE, 0);
    const int8_t* in_ptr = reinterpret_cast<const int8_t*>(inputs.data());
    const int8_t* wt_ptr = reinterpret_cast<const int8_t*>(weights.data());
    for (int i = 0; i < MAT_SIZE; ++i) {
        for (int j = 0; j < MAT_SIZE; ++j) {
            int32_t sum = 0;
            for (int k = 0; k < MAT_SIZE; ++k) {
                int8_t a = in_ptr[i * MAT_SIZE + k];
                int8_t b = wt_ptr[k * MAT_SIZE + j];
                sum += static_cast<int32_t>(a) * static_cast<int32_t>(b);
            }
            results_32bit[i * MAT_SIZE + j] = sum;
        }
    }
    std::vector<uint8_t> results_bytes(MAT_SIZE * MAT_SIZE * sizeof(int32_t));
    std::memcpy(results_bytes.data(), results_32bit.data(), results_bytes.size());
    return results_bytes;
}

Accumulator::Accumulator(size_t entries) : size(entries), state(CompState::IDLE), cycles_remaining(0) {}

void Accumulator::tick() {
    if (state == CompState::BUSY) {
        cycles_remaining--;
        if (cycles_remaining <= 0) {
            switch (pending_op) {
                case AccOp::WRITE:    write_internal();    break;
                case AccOp::READ:     read_internal();     break;
                case AccOp::ACTIVATE: activate_internal(); break;
            }
            state = CompState::IDLE;
        }
    }
}

bool Accumulator::write_request(uint32_t addr, const std::vector<uint8_t>& data) {
    if (state == CompState::BUSY) return false;
    state = CompState::BUSY;
    cycles_remaining = LATENCY_ACC_WRITE;
    pending_op = AccOp::WRITE;
    op_addr = addr;
    write_data_buffer = data;
    return true;
}

bool Accumulator::read_request(uint32_t addr, uint32_t length) {
    if (state == CompState::BUSY) return false;
    state = CompState::BUSY;
    cycles_remaining = LATENCY_ACC_READ;
    pending_op = AccOp::READ;
    op_addr = addr;
    op_length_or_elements = length;
    return true;
}

bool Accumulator::activate_request(uint32_t addr, uint32_t num_elements) {
    if (state == CompState::BUSY) return false;
    state = CompState::BUSY;
    cycles_remaining = LATENCY_ACTIVATE;
    pending_op = AccOp::ACTIVATE;
    op_addr = addr;
    op_length_or_elements = num_elements;
    return true;
}

std::vector<uint8_t> Accumulator::get_read_result() {
    return this->read_result_buffer;
}

void Accumulator::write_internal() {
    for (size_t i = 0; i < write_data_buffer.size(); ++i) { 
        this->memory[op_addr + i] = write_data_buffer[i]; 
    }
}

void Accumulator::read_internal() {
    read_result_buffer.clear();
    read_result_buffer.resize(op_length_or_elements, 0);
    for (uint32_t i = 0; i < op_length_or_elements; ++i) {
        auto it = this->memory.find(op_addr + i);
        if (it != this->memory.end()) { read_result_buffer[i] = it->second; }
    }
}

void Accumulator::activate_internal() {
    uint32_t num_elements = op_length_or_elements;
    uint32_t length_bytes = num_elements * sizeof(int32_t);
    std::vector<uint8_t> data_bytes(length_bytes, 0);
    for (uint32_t i = 0; i < length_bytes; ++i) {
        auto it = this->memory.find(op_addr + i);
        if (it != this->memory.end()) { data_bytes[i] = it->second; }
    }
    std::vector<int32_t> elements(num_elements);
    std::memcpy(elements.data(), data_bytes.data(), length_bytes);
    for (uint32_t i = 0; i < num_elements; ++i) {
        if (elements[i] < 0) {
            elements[i] = 0;
        }
    }
    std::memcpy(data_bytes.data(), elements.data(), length_bytes);
    for (size_t i = 0; i < length_bytes; ++i) { 
        this->memory[op_addr + i] = data_bytes[i]; 
    }
}

void Accumulator::write(uint32_t addr, const std::vector<uint8_t>& data) {
    for (size_t i = 0; i < data.size(); ++i) this->memory[addr + i] = data[i];
}
std::vector<uint8_t> Accumulator::read(uint32_t addr, uint32_t length) {
    std::vector<uint8_t> data_out(length, 0);
    for (uint32_t i = 0; i < length; ++i) {
        auto it = this->memory.find(addr + i);
        if (it != this->memory.end()) data_out[i] = it->second;
    }
    return data_out;
}
void Accumulator::activate(uint32_t addr, uint32_t num_elements) {
    activate_internal();
}
