/**
 * @file main_web.cpp
 * @brief WebAssembly Entry Point
 *
 * This file serves as the main entry point for the Emscripten-compiled WebAssembly module.
 * It exposes the simulation runners to JavaScript, allowing the browser to execute
 * the high-performance C++ double pendulum solvers.
 */
#include "simulation.hpp"
#include <emscripten.h>

extern "C" {
    int main() {
        run_naive_rk4_simulation("pendulum.csv", false);
        run_adaptive_rk4_simulation("adaptive_step_size_pendulum.csv", false);
        run_boost_rkd5_simulation("boost_pendulum.csv", false);
        return 0;
    }
}
