#include "volterra/options.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    return values[values.size()/2];
}
}

int main() {
    using namespace volterra;
    constexpr std::size_t repeats=7;
    constexpr std::size_t analytic_count=1'000'000;
    std::vector<double> analytic_seconds;
    volatile double checksum=0.0;
    Contract c{OptionType::call,ExerciseStyle::european,100,105,.5,.04,.01,.25};
    for(std::size_t run=0;run<repeats+1;++run) {
        const auto start=clock_type::now();
        double sum=0.0;
        for(std::size_t i=0;i<analytic_count;++i) {
            c.spot=95.0+static_cast<double>(i%10000)*.001;
            sum+=black_scholes(c).price;
        }
        const auto elapsed=std::chrono::duration<double>(clock_type::now()-start).count();
        checksum=checksum+sum;
        if(run>0) analytic_seconds.push_back(elapsed);
    }
    const double bs_median=median(analytic_seconds);
    std::cout<<std::fixed<<std::setprecision(3)
      <<"Black-Scholes: "<<analytic_count<<" varied contracts; median "<<bs_median*1000.0<<" ms; "
      <<(static_cast<double>(analytic_count)/bs_median)<<" prices/s; "
      <<(bs_median*1e9/static_cast<double>(analytic_count))<<" ns/price\n";

    std::vector<double> tree_us;
    double tree_price=0.0;
    auto american=c; american.type=OptionType::put; american.style=ExerciseStyle::american;
    for(std::size_t i=0;i<11;++i) {
        const auto start=clock_type::now();
        tree_price=binomial_price(american,512);
        const double elapsed=std::chrono::duration<double,std::micro>(clock_type::now()-start).count();
        if(i>0) tree_us.push_back(elapsed);
    }
    std::cout<<"CRR American put: 512 steps; median "<<median(tree_us)<<" us; price "<<tree_price<<'\n';

    std::vector<double> mc_ms;
    MonteCarloResult mc{};
    for(std::size_t i=0;i<5;++i) {
        const auto start=clock_type::now();
        mc=monte_carlo_price(c,250'000,2026);
        const double elapsed=std::chrono::duration<double,std::milli>(clock_type::now()-start).count();
        if(i>0) mc_ms.push_back(elapsed);
    }
    std::cout<<"Antithetic Monte Carlo: "<<mc.paths<<" paths; median "<<median(mc_ms)
      <<" ms; price "<<mc.price<<"; standard_error "<<mc.standard_error<<'\n';

    std::vector<double> lsm_ms;
    MonteCarloResult lsm{};
    auto american_put=c; american_put.type=OptionType::put; american_put.style=ExerciseStyle::american;
    for(std::size_t i=0;i<5;++i) {
        const auto start=clock_type::now();
        lsm=american_monte_carlo_price(american_put,50'000,50,2026);
        const double elapsed=std::chrono::duration<double,std::milli>(clock_type::now()-start).count();
        if(i>0) lsm_ms.push_back(elapsed);
    }
    std::cout<<"Longstaff-Schwartz American put: "<<lsm.paths<<" paths x 50 steps; median "
      <<median(lsm_ms)<<" ms; price "<<lsm.price<<"; standard_error "<<lsm.standard_error<<'\n';
    std::cout<<"checksum="<<checksum<<"\n";
}
