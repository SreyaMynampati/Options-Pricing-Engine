#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace volterra {

enum class OptionType { call, put };
enum class ExerciseStyle { european, american };

struct Contract {
    OptionType type{OptionType::call};
    ExerciseStyle style{ExerciseStyle::european};
    double spot{};
    double strike{};
    double time_to_expiry{}; // Years, using the caller's day-count convention.
    double rate{};           // Continuously compounded annual risk-free rate.
    double dividend_yield{}; // Continuously compounded annual yield.
    double volatility{};     // Annualized decimal volatility (0.20 = 20%).
};

struct PriceGreeks {
    double price{};
    double delta{};
    double gamma{};
    double vega{}; // Per 1.00 volatility, not per volatility point.
    double theta{}; // Calendar-time decay per year.
    double rho{};   // Per 1.00 rate.
};

struct ImpliedVolResult {
    double volatility{};
    double model_price{};
    double price_error{};
    std::size_t iterations{};
    bool converged{};
    std::string_view method{};
};

struct MonteCarloResult {
    double price{};
    double standard_error{};
    double confidence_low{};
    double confidence_high{};
    std::uint64_t paths{};
};

// Validates finite inputs and economically meaningful contract bounds.
void validate(const Contract& contract);

// European analytical price and first/second order Greeks. Rejects American contracts.
[[nodiscard]] PriceGreeks black_scholes(const Contract& contract);

// Cox-Ross-Rubinstein tree with early exercise for American contracts.
[[nodiscard]] double binomial_price(const Contract& contract, std::size_t steps = 512);

// Reproducible antithetic Monte Carlo for European terminal-payoff contracts.
// Confidence limits use a normal approximation; increase paths for tail-sensitive work.
[[nodiscard]] MonteCarloResult monte_carlo_price(
    const Contract& contract, std::uint64_t paths = 1'000'000, std::uint64_t seed = 42);

// Longstaff-Schwartz least-squares Monte Carlo for American exercise.
// Uses antithetic GBM paths and a quadratic continuation basis in log-moneyness.
[[nodiscard]] MonteCarloResult american_monte_carlo_price(
    const Contract& contract, std::uint64_t paths = 100'000,
    std::size_t time_steps = 50, std::uint64_t seed = 42);

// Safeguarded Newton solver: Newton steps are accepted only inside a maintained bracket;
// otherwise a bisection step is used. European contracts only.
[[nodiscard]] std::optional<ImpliedVolResult> implied_volatility(
    Contract contract, double market_price, double price_tolerance = 1e-10,
    double volatility_tolerance = 1e-10, std::size_t max_iterations = 100);

} // namespace volterra
