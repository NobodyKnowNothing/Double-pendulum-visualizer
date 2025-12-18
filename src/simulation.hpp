#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

// Simulation parameters
const double g = 9.81;
const double m1 = 2;
const double m2 = 1;
const double l1 = 2;
const double l2 = 1;

// Time parameters
const double t_start = 0.0;
const double t_end = 40.0;
const double steps = 1001;
const double dt = (t_end - t_start) / (steps - 1);

struct State {
    double th1;
    double w1;
    double th2;
    double w2;
};

// Calculate derivatives (Equations of motion)
inline State compute_derivatives(const State& s) {
    double delta = s.th1 - s.th2;
    double den = 2 * m1 + m2 - m2 * std::cos(2* s.th1 - 2 * s.th2);

    double num1 = -g * (2 * m1 + m2) * std::sin(s.th1) - m2 * g * std::sin(s.th1 - 2 * s.th2) - 2 * std::sin(delta) * m2 * (s.w2 * s.w2 * l2 + s.w1 * s.w1 * l1 * std::cos(delta));
    double alpha1 = num1 / (l1 * den);

    double num2 = 2 * std::sin(delta) * (s.w1 * s.w1 * l1 * (m1 + m2) + g * (m1 + m2) * std::cos(s.th1) + s.w2 * s.w2 * l2 * m2 * std::cos(delta));
    double alpha2 = num2 / (l2 * den);

    return State{s.w1, alpha1, s.w2, alpha2};
}

// Helper function to add states
inline State add_states(const State& a, const State& b, double scale) {
    return { 
        a.th1 + b.th1 * scale,
        a.w1 + b.w1 * scale,
        a.th2 + b.th2 * scale,
        a.w2 + b.w2 * scale
    };
}

// Runge-Kutta 4 Integrator
inline State rk4_step(const State& current, double step_size) {
    State k1 = compute_derivatives(current);
    State k2 = compute_derivatives(add_states(current, k1, step_size / 2.0));
    State k3 = compute_derivatives(add_states(current, k2, step_size / 2.0));
    State k4 = compute_derivatives(add_states(current, k3, step_size));
    
    State next;
    next.th1 = current.th1 + (step_size / 6.0) * (k1.th1 + 2 * k2.th1 + 2 * k3.th1 + k4.th1);
    next.w1 = current.w1 + (step_size / 6.0) * (k1.w1 + 2 * k2.w1 + 2 * k3.w1 + k4.w1);
    next.th2 = current.th2 + (step_size / 6.0) * (k1.th2 + 2 * k2.th2 + 2 * k3.th2 + k4.th2);
    next.w2 = current.w2 + (step_size / 6.0) * (k1.w2 + 2 * k2.w2 + 2 * k3.w2 + k4.w2);
    
    return next;
}

inline void run_simulation(const std::string& filename, bool verbose = true) {
    // init cond: [theta1, w1, theta2, w2]
    // Matches python y0[1, -3, -1, 5]
    State state = {1.0, -3.0, -1.0, 5.0};

    // Fixed step size
    std::ofstream file(filename);
    if (!file.is_open()) {
        if (verbose) std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    file << "t,th1,w1,th2,w2\n";

    double t = t_start;
    
    if (verbose) std::cout << "Simulating fixed step size double pendulum..." << std::endl;

    for (int i = 0; i <= steps; ++i) {
        // Write to file
        file << t << "," << state.th1 << "," << state.w1 << "," << state.th2 << "," << state.w2 << "\n";
        
        state = rk4_step(state, dt);
        t += dt;
    }

    file.close();

    if (verbose) std::cout << "Simulation complete. Data saved to " << filename << std::endl;

    // Adaptive step size
    std::string filename1 = "adaptive_step_size_" + filename;

    if (verbose) std::cout << "Simulating adaptive step size..." << std::endl;

    file.open(filename1);
    if (!file.is_open()) {
        if (verbose) std::cerr << "Error: Could not open file " << filename1 << std::endl;
        return;
    }

    file << "t,th1,w1,th2,w2\n";
    // Reset init cond
    state = {1.0, -3.0, -1.0, 5.0};
    t = t_start;
    double step_size = dt;
    double error_tolerance = 1e-6;
    double error;
    
    if (verbose) std::cout << "Simulating adaptive step size..." << std::endl;
    State full_step;
    State partial_step;
    int n = 2; // For later experimentation

    file << t << "," << state.th1 << "," << state.w1 << "," << state.th2 << "," << state.w2 << "\n";

    while (t < t_end) {
        // Write to file
        
        full_step = rk4_step(state, step_size);

        partial_step = rk4_step(state, step_size/n);
        for (int i = 0; i < n - 1; ++i) partial_step = rk4_step(partial_step, step_size/n);
        
        error = std::max({
            std::abs(full_step.th1 - partial_step.th1), 
            std::abs(full_step.w1 - partial_step.w1), 
            std::abs(full_step.th2 - partial_step.th2), 
            std::abs(full_step.w2 - partial_step.w2)
        });
        
        if (error > error_tolerance) {
            step_size *= 0.5;
        } else {
            t += step_size;
            state = partial_step;
            file << t << "," << state.th1 << "," << state.w1 << "," << state.th2 << "," << state.w2 << "\n";
            if (error < error_tolerance/10 && step_size < dt) step_size *= 2;
        }
    }

    file.close();
    if (verbose) std::cout << "Adaptive simulation complete. Data saved to " << filename1 << std::endl;
}

#endif
