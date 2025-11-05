#include "tpu.h"
#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "--- Running Python Compiler ---" << std::endl;
    int compiler_ret = std::system("python3 compiler.py");
    if (compiler_ret != 0) {
        std::cerr << "FATAL: Python compiler script failed!" << std::endl;
        return 1;
    }
    std::cout << "--- Compiler finished ---\n" << std::endl;
    
    TPU my_tpu;
    
    my_tpu.load_program("program.bin");
    my_tpu.load_host_memory("memory.bin");

    std::cout << "\n--- RUNNING CYCLE-ACCURATE SIMULATION ---" << std::endl;
    
    while (!my_tpu.is_halted()) {
        my_tpu.tick();
        
        if (my_tpu.get_cycle_count() > 5000000) { 
            std::cout << "ERROR: Simulation timed out!" << std::endl;
            break;
        }
    }
    
    std::cout << "--- SIMULATION HALTED ---" << std::endl;
    
    my_tpu.print_performance_report();
    
    return 0;
}
