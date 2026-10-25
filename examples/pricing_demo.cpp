#include "volterra/options.hpp"
#include <iomanip>
#include <iostream>

int main() {
    using namespace volterra;
    Contract call{OptionType::call, ExerciseStyle::european, 100.0, 105.0, 0.5, 0.04, 0.01, 0.25};
    const auto bs=black_scholes(call);
    auto american=call; american.type=OptionType::put; american.style=ExerciseStyle::american;
    const double tree=binomial_price(american,1000);
    const auto mc=monte_carlo_price(call,500'000,2026);
    call.volatility=0.0;
    const auto iv=implied_volatility(call,bs.price);
    std::cout<<std::fixed<<std::setprecision(6)
      <<"European call (Black-Scholes): "<<bs.price<<"\n"
      <<"European call delta / gamma / vega: "<<bs.delta<<" / "<<bs.gamma<<" / "<<bs.vega<<"\n"
      <<"American put (CRR tree): "<<tree<<"\n"
      <<"European call (MC, 95% CI): "<<mc.price<<" ["<<mc.confidence_low<<", "<<mc.confidence_high<<"]\n"
      <<"Recovered implied volatility: "<<(iv?iv->volatility:0.0)<<"\n";
}
