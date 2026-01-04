int main() {
    run_naive_rk4_simulation("pendulum0.csv", true);
    run_adaptive_rk4_simulation("pendulum1.csv", true);
    run_boost_rkd5_simulation("pendulum2.csv", true);
    return 0;
}
