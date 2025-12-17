#include <iostream>
#include <cmath>
#include <emscripten.h>
#include <vector>
#include <fstream>

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
// These are analytical solutions that SymPy solves for in the python file
State compute_derivatives(const State& s) {
    double delta = s.th1 - s.th2;
    double den = 2 * m1 + m2 - m2 * std::cos(2* s.th1 - 2 * s.th2);

    double num1 = -g * (2 * m1 + m2) * std::sin(s.th1) - m2 * g * std::sin(s.th1 - 2 * s.th2) - 2 * std::sin(delta) * m2 * (s.w2 * s.w2 * l2 + s.w1 * s.w1 * l1 * std::cos(delta));
    double alpha1 = num1 / (l1 * den);

    double num2 = 2 * std::sin(delta) * (s.w1 * s.w1 * l1 * (m1 + m2) + g * (m1 + m2) * std::cos(s.th1) + s.w2 * s.w2 * l2 * m2 * std::cos(delta));
    double alpha2 = num2 / (l2 * den);

    return State{s.w1, alpha1, s.w2, alpha2};
}

// Helper function to add states
State add_states(const State& a, const State& b, double scale) {
    return { 
        a.th1 + b.th1 * scale,
        a.w1 + b.w1 * scale,
        a.th2 + b.th2 * scale,
        a.w2 + b.w2 * scale
    };
}

// Runge-Kutta 4 Intergrator
State rk4_step(const State& current) {
    State k1 = compute_derivatives(current);
    State k2 = compute_derivatives(add_states(current, k1, dt / 2.0));
    State k3 = compute_derivatives(add_states(current, k2, dt / 2.0));
    State k4 = compute_derivatives(add_states(current, k3, dt));
    
    State next;
    next.th1 = current.th1 + (dt / 6.0) * (k1.th1 + 2 * k2.th1 + 2 * k3.th1 + k4.th1);
    next.w1 = current.w1 + (dt / 6.0) * (k1.w1 + 2 * k2.w1 + 2 * k3.w1 + k4.w1);
    next.th2 = current.th2 + (dt / 6.0) * (k1.th2 + 2 * k2.th2 + 2 * k3.th2 + k4.th2);
    next.w2 = current.w2 + (dt / 6.0) * (k1.w2 + 2 * k2.w2 + 2 * k3.w2 + k4.w2);
    
    return next;
}

extern "C" {
    int main() {
        // init cond: [theta1, w1, theta2, w2]
        // Matches python y0[1, -3, -1, 5]
        State state = {1.0, -3.0, -1.0, 5.0};

        std::ofstream file("pendulum.csv");
        file << "t,th1,w1,th2,w2\n";

        double t = t_start;
        
        // std::cout << "Simulating double pendulum..." << std::endl;


        for (int i = 0; i <= steps; ++i) {
            // Calculate cartesian coordinates for output
            double x1 = l1 * std::sin(state.th1);
            double y1 = -l1 * std::cos(state.th1);
            double x2 = x1 + l2 * std::sin(state.th2);
            double y2 = y1 - l2 * std::cos(state.th2);

            // Write to file
            file << t << "," << state.th1 << "," << state.w1 << "," << state.th2 << "," << state.w2 << "\n";
            
            state = rk4_step(state);
            t += dt;
        }

        file.close();
        // std::cout << "Simulation complete. Data saved to pendulum.csv" << std::endl;
        
        return 0;
    }
}
