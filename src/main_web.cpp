#include "simulation.hpp"
#include <emscripten.h>

extern "C" {
    int main() {
        run_simulation("pendulum.csv", false);
        return 0;
    }
}
