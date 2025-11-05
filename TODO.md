# Revamp Codebase Plan

## Information Gathered
- The codebase is a TPU simulator written in C++ with a Python compiler.
- It includes classes for TPU components (UnifiedBuffer, SystolicArray, etc.), ISA definitions, and a main simulation loop.
- Functionality: Cycle-accurate simulation of TPU operations like matrix multiply, activation, memory transfers.
- Output: Performance report and simulation results must remain identical.

## Plan
- Make code look messy like a student's personal project: inconsistent formatting (mix tabs/spaces, random indentation), poor variable/function names (e.g., 'x' instead of 'addr'), unhelpful comments (e.g., "// stuff"), spaghetti code (long functions, mixed logic), but keep functionality intact.
- Edit files one by one: isa.h, tpu_components.h, tpu_components.cpp, tpu.h, tpu.cpp, main.cpp, compiler.py.
- For each file, introduce messiness without changing logic: rename variables, add dummy vars, inconsistent spacing, useless comments.
- Ensure no syntax errors; code must compile and run identically.

## Dependent Files to Edit
- isa.h
- tpu_components.h
- tpu_components.cpp
- tpu.h
- tpu.cpp
- main.cpp
- compiler.py

## Followup Steps
- After editing all files, run the simulation (./tpu_sim or equivalent) to verify output matches original.
- If issues, debug and fix while keeping messiness.
