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
struct LStates {
    struct State l_s1 {1.0, 0.0, 0.0, 0.0};
    struct State l_s2 {0.0, 1.0, 0.0, 0.0};
    struct State l_s3 {0.0, 0.0, 1.0, 0.0};
    struct State l_s4 {0.0, 0.0, 0.0, 1.0};
    double log_sum1 = 0;
    double log_sum2 = 0;
    double log_sum3 = 0;
    double log_sum4 = 0;
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
inline State add_states(const State& a, const State& b, double scale = 1.0) {
    return { 
        a.th1 + b.th1 * scale,
        a.w1 + b.w1 * scale,
        a.th2 + b.th2 * scale,
        a.w2 + b.w2 * scale
    };
}

// Helper to subtract states
inline State subtract_states(const State& a, const State& b, double scale = 1.0) {
    return { 
        a.th1 - b.th1 * scale,
        a.w1 - b.w1 * scale,
        a.th2 - b.th2 * scale,
        a.w2 - b.w2 * scale
    };
}

inline LStates add_lstates(const LStates& a, const LStates& b, double scale) {
    return { 
        add_states(a.l_s1, b.l_s1, scale),
        add_states(a.l_s2, b.l_s2, scale),
        add_states(a.l_s3, b.l_s3, scale),
        add_states(a.l_s4, b.l_s4, scale),
        a.log_sum1, a.log_sum2, a.log_sum3, a.log_sum4
    };
}

inline State multiply_state(const State& a, double scale) {
    return { 
        a.th1 * scale,
        a.w1 * scale,
        a.th2 * scale,
        a.w2 * scale
    };
}

inline State compute_sensitivity(const State& real, const State& fake, const double epsilon = 1e-6) {
    return multiply_state(subtract_states(compute_derivatives(add_states(real, fake, epsilon)), compute_derivatives(real)), 1/epsilon);
}

// Lyapunov Exponent
inline LStates fake_derivatives(const State& real, const LStates& fake) {
    return {
        compute_sensitivity(real, fake.l_s1),
        compute_sensitivity(real, fake.l_s2),
        compute_sensitivity(real, fake.l_s3),
        compute_sensitivity(real, fake.l_s4)
    };
}

inline double state_mag(const State& s) {
    return std::sqrt(s.th1*s.th1 + s.w1*s.w1 + s.th2*s.th2 + s.w2*s.w2);
}

inline double dot_product(const State& a, const State& b) {
    return (a.th1*b.th1 + a.w1*b.w1 + a.th2*b.th2 + a.w2*b.w2);
}

inline void gram_shmidt_shuffle(LStates& fake) {
    double mag1 = state_mag(fake.l_s1);
    fake.log_sum1 += std::log(mag1);

    fake.l_s1 = multiply_state(fake.l_s1, 1.0/mag1);

    double overlap12 = dot_product(fake.l_s2, fake.l_s1);
    fake.l_s2 = subtract_states(fake.l_s2, multiply_state(fake.l_s1, overlap12));
    
    double mag2 = state_mag(fake.l_s2);
    fake.log_sum2 += std::log(mag2);
    fake.l_s2 = multiply_state(fake.l_s2, 1.0/mag2);

    double overlap13 = dot_product(fake.l_s3, fake.l_s1);
    double overlap23 = dot_product(fake.l_s3, fake.l_s2);
    
    fake.l_s3 = subtract_states(fake.l_s3, multiply_state(fake.l_s1, overlap13));
    fake.l_s3 = subtract_states(fake.l_s3, multiply_state(fake.l_s2, overlap23));

    double mag3 = state_mag(fake.l_s3);
    fake.log_sum3 += std::log(mag3);
    fake.l_s3 = multiply_state(fake.l_s3, 1.0/mag3);

    double overlap14 = dot_product(fake.l_s4, fake.l_s1);
    double overlap24 = dot_product(fake.l_s4, fake.l_s2);
    double overlap34 = dot_product(fake.l_s4, fake.l_s3);

    fake.l_s4 = subtract_states(fake.l_s4, multiply_state(fake.l_s1, overlap14));
    fake.l_s4 = subtract_states(fake.l_s4, multiply_state(fake.l_s2, overlap24));
    fake.l_s4 = subtract_states(fake.l_s4, multiply_state(fake.l_s3, overlap34));
    
    double mag4 = state_mag(fake.l_s4);
    fake.log_sum4 += std::log(mag4);
    fake.l_s4 = multiply_state(fake.l_s4, 1.0/mag4);
}

inline LStates rk4_lyanpunov_step(const LStates& fake, const State& current, double step_size) {
    State rk1 = compute_derivatives(current);
    State rmid1 = add_states(current, rk1, step_size / 2.0);
    State rk2 = compute_derivatives(rmid1);
    State rmid2 = add_states(current, rk2, step_size / 2.0);
    State rk3 = compute_derivatives(rmid2);
    State rmid3 = add_states(current, rk3, step_size);

    LStates k1 = fake_derivatives(current, fake);
    LStates k2 = fake_derivatives(rmid1, add_lstates(fake, k1, step_size / 2.0));
    LStates k3 = fake_derivatives(rmid2, add_lstates(fake, k2, step_size / 2.0));
    LStates k4 = fake_derivatives(rmid3, add_lstates(fake, k3, step_size));
    
    LStates next = fake;
    next.l_s1.th1 = fake.l_s1.th1 + (step_size / 6.0) * (k1.l_s1.th1 + 2 * k2.l_s1.th1 + 2 * k3.l_s1.th1 + k4.l_s1.th1);
    next.l_s1.w1 = fake.l_s1.w1 + (step_size / 6.0) * (k1.l_s1.w1 + 2 * k2.l_s1.w1 + 2 * k3.l_s1.w1 + k4.l_s1.w1);
    next.l_s1.th2 = fake.l_s1.th2 + (step_size / 6.0) * (k1.l_s1.th2 + 2 * k2.l_s1.th2 + 2 * k3.l_s1.th2 + k4.l_s1.th2);
    next.l_s1.w2 = fake.l_s1.w2 + (step_size / 6.0) * (k1.l_s1.w2 + 2 * k2.l_s1.w2 + 2 * k3.l_s1.w2 + k4.l_s1.w2);

    next.l_s2.th1 = fake.l_s2.th1 + (step_size / 6.0) * (k1.l_s2.th1 + 2 * k2.l_s2.th1 + 2 * k3.l_s2.th1 + k4.l_s2.th1);
    next.l_s2.w1 = fake.l_s2.w1 + (step_size / 6.0) * (k1.l_s2.w1 + 2 * k2.l_s2.w1 + 2 * k3.l_s2.w1 + k4.l_s2.w1);
    next.l_s2.th2 = fake.l_s2.th2 + (step_size / 6.0) * (k1.l_s2.th2 + 2 * k2.l_s2.th2 + 2 * k3.l_s2.th2 + k4.l_s2.th2);
    next.l_s2.w2 = fake.l_s2.w2 + (step_size / 6.0) * (k1.l_s2.w2 + 2 * k2.l_s2.w2 + 2 * k3.l_s2.w2 + k4.l_s2.w2);

    next.l_s3.th1 = fake.l_s3.th1 + (step_size / 6.0) * (k1.l_s3.th1 + 2 * k2.l_s3.th1 + 2 * k3.l_s3.th1 + k4.l_s3.th1);
    next.l_s3.w1 = fake.l_s3.w1 + (step_size / 6.0) * (k1.l_s3.w1 + 2 * k2.l_s3.w1 + 2 * k3.l_s3.w1 + k4.l_s3.w1);
    next.l_s3.th2 = fake.l_s3.th2 + (step_size / 6.0) * (k1.l_s3.th2 + 2 * k2.l_s3.th2 + 2 * k3.l_s3.th2 + k4.l_s3.th2);
    next.l_s3.w2 = fake.l_s3.w2 + (step_size / 6.0) * (k1.l_s3.w2 + 2 * k2.l_s3.w2 + 2 * k3.l_s3.w2 + k4.l_s3.w2);

    next.l_s4.th1 = fake.l_s4.th1 + (step_size / 6.0) * (k1.l_s4.th1 + 2 * k2.l_s4.th1 + 2 * k3.l_s4.th1 + k4.l_s4.th1);
    next.l_s4.w1 = fake.l_s4.w1 + (step_size / 6.0) * (k1.l_s4.w1 + 2 * k2.l_s4.w1 + 2 * k3.l_s4.w1 + k4.l_s4.w1);
    next.l_s4.th2 = fake.l_s4.th2 + (step_size / 6.0) * (k1.l_s4.th2 + 2 * k2.l_s4.th2 + 2 * k3.l_s4.th2 + k4.l_s4.th2);
    next.l_s4.w2 = fake.l_s4.w2 + (step_size / 6.0) * (k1.l_s4.w2 + 2 * k2.l_s4.w2 + 2 * k3.l_s4.w2 + k4.l_s4.w2);

    return next;
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
    State state = {1.0, -3.0, -1.0, 5.0};
    LStates ghosts;

    std::ofstream file(filename);
    if (!file.is_open()) {
        if (verbose) std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    file << "t,th1,w1,th2,w2,l1,l2,l3,l4\n";

    double t = t_start;
    
    if (verbose) std::cout << "Simulating fixed step size double pendulum..." << std::endl;

    for (int i = 0; i <= steps; ++i) {
        // Write to file
        file << t << "," << state.th1 << "," << state.w1 << "," << state.th2 << "," << state.w2 << "," << ghosts.log_sum1 << "," << ghosts.log_sum2 << "," << ghosts.log_sum3 << "," << ghosts.log_sum4 << "\n";
        ghosts = rk4_lyanpunov_step(ghosts, state, dt);
        state = rk4_step(state, dt);
        gram_shmidt_shuffle(ghosts);
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

    file << "t,th1,w1,th2,w2,l1,l2,l3,l4\n";
    // Reset init cond
    LStates ghosts1;
    ghosts = ghosts1;
    state = {1.0, -3.0, -1.0, 5.0};
    t = t_start;
    double step_size = dt;
    double error_tolerance = 1e-6;
    double error;
    
    if (verbose) std::cout << "Simulating adaptive step size..." << std::endl;
    State full_step;
    State partial_step;
    int n = 2; // For later experimentation

    file << t << "," << state.th1 << "," << state.w1 << "," << state.th2 << "," << state.w2 << "," << ghosts.log_sum1 << "," << ghosts.log_sum2 << "," << ghosts.log_sum3 << "," << ghosts.log_sum4 << "\n";

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
            ghosts = rk4_lyanpunov_step(ghosts, state, step_size);
            state = partial_step;
            file << t << "," << state.th1 << "," << state.w1 << "," << state.th2 << "," << state.w2 << "," << ghosts.log_sum1 << "," << ghosts.log_sum2 << "," << ghosts.log_sum3 << "," << ghosts.log_sum4 << "\n";
            gram_shmidt_shuffle(ghosts);
            if (error < error_tolerance/10 && step_size < dt) step_size *= 2;
        }
    }

    file.close();
    if (verbose) std::cout << "Adaptive simulation complete. Data saved to " << filename1 << std::endl;
}

#endif
