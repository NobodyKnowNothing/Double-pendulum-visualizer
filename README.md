# Double Pendulum Visualizer

This project simulates the chaotic motion of a double pendulum using both Python and C++. It demonstrates the performance difference between a purely Python-based solver and two kinds of C++ Runge-Kutta solvers compiled to WebAssembly (WASM), visualized in the browser.

## Features

- **Python Simulation**: Uses `scipy.integrate.odeint` to solve the Lagrangian equations of motion. Visualized using `matplotlib`.
- **Web Simulation**: Compare Python (PyScript) and C++ (WASM) performance side-by-side in the browser.
- **Performance Comparison**: Real-time visualization of the speed difference between interpreted Python and compiled C++ WASM.

## Project Structure

- `src/cpp/`: C++ source code for the two Runge-Kutta solvers.
- `src/python/`: Python source code for the desktop simulation and visualization.
- `web/`: The web application containing the HTML, WASM binaries, and PyScript logic.
- `scripts/`: various helper scripts.
- `assets/`: Images and other static assets.

## Usage

### 1. Desktop Python Simulation

To run the double pendulum simulation locally using Python:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
python src/python/main.py
```

This will solve the equations of motion and display an animation of the double pendulum.

### 2. Web Visualization (WASM vs Python)

To run the web comparison:

1.  Start a local HTTP server in the root directory:
    ```bash
    python -m http.server
    ```
2.  Open your browser and navigate to:
    [http://localhost:8000/web/](http://localhost:8000/web/)

You will see the double pendulum simulation running. The stats panel will show the performance metrics for both the C++ WASM solver and the Python solver.

## Building from Source (C++ to WASM)

If you wish to modify the C++ solver and recompile it to WASM, you will need the [Emscripten SDK (emsdk)](https://emscripten.org/docs/getting_started/downloads.html).

1.  Install and activate Emscripten.
2.  Run the compilation command:
    ```bash
    emcc src/cpp/main_web.cpp -O3 -s WASM=1 -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']" -s "EXPORTED_FUNCTIONS=['_createCppSolver', '_malloc', '_free']" -o web/cpp_solver.js
    ```
    *Note: The exact export flags may vary based on your specific modifications.*

## License

[MIT License](LICENSE)
