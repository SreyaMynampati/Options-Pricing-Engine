#include "volterra/options.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool condition, const std::string& name) {
    if (!condition) { std::cerr << "FAIL: " << name << '\n'; ++failures; }
}
void close(double actual, double expected, double tolerance, const std::string& name) {
    check(std::isfinite(actual) && std::abs(actual-expected) <= tolerance, name +
          " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) + ")");
}
volterra::Contract contract(volterra::OptionType type=volterra::OptionType::call,
                            volterra::ExerciseStyle style=volterra::ExerciseStyle::european,
                            double s=100, double k=100, double t=1, double r=.05,
                            double q=0, double v=.2) {
    return {type,style,s,k,t,r,q,v};
}
}

int main() {
    using namespace volterra;
    const auto call=black_scholes(contract());
    const auto put=black_scholes(contract(OptionType::put));
    close(call.price,10.450583572185565,2e-10,"Black-Scholes reference call");
    close(put.price,5.573526022256971,2e-10,"Black-Scholes reference put");
    close(call.price-put.price,100.0-100.0*std::exp(-.05),2e-12,"put-call parity");

    const auto qcall=contract(OptionType::call,ExerciseStyle::european,103,97,.7,.037,.012,.31);
    const auto qc=black_scholes(qcall);
    auto qp=qcall; qp.type=OptionType::put;
    const auto qput=black_scholes(qp);
    close(qc.price-qput.price,qcall.spot*std::exp(-qcall.dividend_yield*qcall.time_to_expiry)-
          qcall.strike*std::exp(-qcall.rate*qcall.time_to_expiry),1e-11,"dividend-adjusted parity");

    const double h=1e-4;
    auto up=qcall, down=qcall; up.spot+=h; down.spot-=h;
    const double delta_fd=(black_scholes(up).price-black_scholes(down).price)/(2*h);
    constexpr double gamma_h=1e-2;
    auto gamma_up=qcall, gamma_down=qcall; gamma_up.spot+=gamma_h; gamma_down.spot-=gamma_h;
    const double gamma_fd=(black_scholes(gamma_up).price-2*qc.price+black_scholes(gamma_down).price)/(gamma_h*gamma_h);
    close(qc.delta,delta_fd,2e-8,"delta finite difference");
    close(qc.gamma,gamma_fd,2e-7,"gamma finite difference");
    up=qcall; down=qcall; up.volatility+=h; down.volatility-=h;
    close(qc.vega,(black_scholes(up).price-black_scholes(down).price)/(2*h),2e-7,"vega finite difference");
    up=qcall; down=qcall; up.rate+=h; down.rate-=h;
    close(qc.rho,(black_scholes(up).price-black_scholes(down).price)/(2*h),2e-7,"rho finite difference");
    up=qcall; down=qcall; up.time_to_expiry+=h; down.time_to_expiry-=h;
    close(qc.theta,-(black_scholes(up).price-black_scholes(down).price)/(2*h),2e-7,"theta finite difference");

    auto zero=contract(OptionType::call,ExerciseStyle::european,100,95,1,.05,.02,0);
    close(black_scholes(zero).price,std::exp(-.05)*std::max(100*std::exp(.03)-95.0,0.0),1e-12,"zero-vol deterministic price");
    auto expiry=contract(OptionType::put,ExerciseStyle::european,90,100,0,.05,0,.2);
    close(black_scholes(expiry).price,10,0,"expiry intrinsic value");
    close(binomial_price(expiry,100),10,0,"expiry tree intrinsic value");

    auto tree_contract=contract(OptionType::call,ExerciseStyle::european,100,105,.5,.04,.01,.25);
    close(binomial_price(tree_contract,2000),black_scholes(tree_contract).price,.025,"CRR converges to Black-Scholes");
    auto american=tree_contract; american.type=OptionType::put; american.style=ExerciseStyle::american;
    auto european_put=american; european_put.style=ExerciseStyle::european;
    check(binomial_price(american,1000)>=binomial_price(european_put,1000),"American put is not below European put");
    auto american_call=contract(OptionType::call,ExerciseStyle::american,100,90,1,.02,.10,0);
    check(binomial_price(american_call,100)>=black_scholes(contract(OptionType::call,ExerciseStyle::european,100,90,1,.02,.10,0)).price,
          "zero-vol American early exercise dominates European value");

    auto mc_contract=contract(OptionType::put,ExerciseStyle::european,100,105,.75,.03,.01,.35);
    const auto mc1=monte_carlo_price(mc_contract,200000,12345);
    const auto mc2=monte_carlo_price(mc_contract,200000,12345);
    close(mc1.price,mc2.price,0,"Monte Carlo seed reproducibility");
    const double analytical=black_scholes(mc_contract).price;
    check(mc1.confidence_low<=analytical && analytical<=mc1.confidence_high,"Monte Carlo 95% interval covers analytic reference");
    check(mc1.standard_error>0 && mc1.paths==200000,"Monte Carlo standard error and path count");

    for (const auto type : {OptionType::call,OptionType::put}) {
        for (double sigma : {.08,.2,.65,1.5}) {
            auto c=contract(type,ExerciseStyle::european,100,type==OptionType::call?112:88,.4,.025,.008,sigma);
            const double market=black_scholes(c).price;
            c.volatility=.4;
            const auto iv=implied_volatility(c,market);
            check(iv.has_value() && iv->converged,"IV solver converges for valid market price");
            if (iv) close(iv->volatility,sigma,2e-8,"IV round-trip");
        }
    }
    auto invalid=contract();
    check(!implied_volatility(invalid,0.0).has_value(),"IV rejects price below no-arbitrage bound");
    check(!implied_volatility(invalid,100.0).has_value(),"IV rejects price at upper bound");

    bool threw=false;
    try { auto bad=contract(); bad.spot=-1; (void)black_scholes(bad); }
    catch (const std::invalid_argument&) { threw=true; }
    check(threw,"invalid contract input rejected");
    threw=false;
    try { auto bad=contract(OptionType::call,ExerciseStyle::american); (void)black_scholes(bad); }
    catch (const std::invalid_argument&) { threw=true; }
    check(threw,"Black-Scholes rejects American style");

    if (failures) { std::cerr << failures << " validation check(s) failed\n"; return EXIT_FAILURE; }
    std::cout << "All numerical validation checks passed.\n";
    return EXIT_SUCCESS;
}
