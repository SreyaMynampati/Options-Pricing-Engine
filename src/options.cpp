#include "volterra/options.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace volterra {
namespace {
constexpr double sqrt_two = 1.4142135623730950488;
constexpr double inv_sqrt_two_pi = 0.39894228040143267794;
double cdf(double x) { return 0.5 * std::erfc(-x / sqrt_two); }
double pdf(double x) { return inv_sqrt_two_pi * std::exp(-0.5 * x * x); }
double payoff(OptionType type, double spot, double strike) {
    return type == OptionType::call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
}

struct Terms { double d1, d2, df_r, df_q, root_t; };
Terms terms(const Contract& c) {
    const double root_t = std::sqrt(c.time_to_expiry);
    const double vsqrt = c.volatility * root_t;
    const double d1 = (std::log(c.spot / c.strike) +
        (c.rate - c.dividend_yield + 0.5 * c.volatility * c.volatility) * c.time_to_expiry) / vsqrt;
    return {d1, d1 - vsqrt, std::exp(-c.rate * c.time_to_expiry),
            std::exp(-c.dividend_yield * c.time_to_expiry), root_t};
}
} // namespace

void validate(const Contract& c) {
    const double values[] = {c.spot, c.strike, c.time_to_expiry, c.rate, c.dividend_yield, c.volatility};
    for (double x : values) if (!std::isfinite(x)) throw std::invalid_argument("contract inputs must be finite");
    if (c.spot <= 0.0 || c.strike <= 0.0) throw std::invalid_argument("spot and strike must be positive");
    if (c.time_to_expiry < 0.0) throw std::invalid_argument("time to expiry cannot be negative");
    if (c.volatility < 0.0) throw std::invalid_argument("volatility cannot be negative");
}

PriceGreeks black_scholes(const Contract& c) {
    validate(c);
    if (c.style != ExerciseStyle::european) throw std::invalid_argument("Black-Scholes supports European exercise only");
    const double intrinsic = payoff(c.type, c.spot, c.strike);
    if (c.time_to_expiry == 0.0 || c.volatility == 0.0) {
        const double forward_spot = c.spot * std::exp((c.rate - c.dividend_yield) * c.time_to_expiry);
        const double discounted = std::exp(-c.rate * c.time_to_expiry) * payoff(c.type, forward_spot, c.strike);
        return {c.time_to_expiry == 0.0 ? intrinsic : discounted, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    const auto t = terms(c);
    const double n1 = cdf(t.d1), n2 = cdf(t.d2), p1 = cdf(-t.d1), p2 = cdf(-t.d2);
    const double carry_spot = c.spot * t.df_q;
    const double discounted_strike = c.strike * t.df_r;
    const double density = pdf(t.d1);
    const double gamma = t.df_q * density / (c.spot * c.volatility * t.root_t);
    const double vega = carry_spot * density * t.root_t;
    if (c.type == OptionType::call) {
        return {carry_spot * n1 - discounted_strike * n2, t.df_q * n1, gamma, vega,
            -carry_spot * density * c.volatility / (2.0 * t.root_t) - c.rate * discounted_strike * n2 + c.dividend_yield * carry_spot * n1,
            c.strike * c.time_to_expiry * t.df_r * n2};
    }
    return {discounted_strike * p2 - carry_spot * p1, t.df_q * (n1 - 1.0), gamma, vega,
        -carry_spot * density * c.volatility / (2.0 * t.root_t) + c.rate * discounted_strike * p2 - c.dividend_yield * carry_spot * p1,
        -c.strike * c.time_to_expiry * t.df_r * p2};
}

double binomial_price(const Contract& c, std::size_t steps) {
    validate(c);
    if (steps == 0) throw std::invalid_argument("binomial steps must be positive");
    if (c.time_to_expiry == 0.0) return payoff(c.type, c.spot, c.strike);
    if (c.volatility == 0.0) {
        if (c.style == ExerciseStyle::european) {
            const double terminal = c.spot * std::exp((c.rate-c.dividend_yield)*c.time_to_expiry);
            return std::exp(-c.rate*c.time_to_expiry)*payoff(c.type,terminal,c.strike);
        }
        double best = payoff(c.type,c.spot,c.strike);
        for (std::size_t i=1;i<=steps;++i) {
            const double t=c.time_to_expiry*static_cast<double>(i)/static_cast<double>(steps);
            best=std::max(best,std::exp(-c.rate*t)*payoff(c.type,c.spot*std::exp((c.rate-c.dividend_yield)*t),c.strike));
        }
        return best;
    }
    const double dt=c.time_to_expiry/static_cast<double>(steps), up=std::exp(c.volatility*std::sqrt(dt)), down=1.0/up;
    const double growth=std::exp((c.rate-c.dividend_yield)*dt), p=(growth-down)/(up-down);
    if (p<0.0 || p>1.0) throw std::invalid_argument("invalid CRR probability; increase steps or use a different tree scheme");
    const double discount=std::exp(-c.rate*dt);
    std::vector<double> values(steps+1);
    for(std::size_t j=0;j<=steps;++j) {
        const double s=c.spot*std::pow(up,static_cast<double>(j))*std::pow(down,static_cast<double>(steps-j));
        values[j]=payoff(c.type,s,c.strike);
    }
    for(std::size_t i=steps;i-->0;) {
        for(std::size_t j=0;j<=i;++j) {
            const double continuation=discount*(p*values[j+1]+(1.0-p)*values[j]);
            if(c.style==ExerciseStyle::american) {
                const double s=c.spot*std::pow(up,static_cast<double>(j))*std::pow(down,static_cast<double>(i-j));
                values[j]=std::max(continuation,payoff(c.type,s,c.strike));
            } else values[j]=continuation;
        }
    }
    return values[0];
}

MonteCarloResult monte_carlo_price(const Contract& c, std::uint64_t paths, std::uint64_t seed) {
    validate(c);
    if(c.style!=ExerciseStyle::european) throw std::invalid_argument("Monte Carlo currently supports European exercise only");
    if(paths<2) throw std::invalid_argument("Monte Carlo requires at least two paths");
    if(paths%2!=0) --paths;
    if(c.time_to_expiry==0.0 || c.volatility==0.0) return {black_scholes(c).price,0.0,black_scholes(c).price,black_scholes(c).price,paths};
    std::mt19937_64 rng(seed); std::normal_distribution<double> normal;
    const double drift=(c.rate-c.dividend_yield-0.5*c.volatility*c.volatility)*c.time_to_expiry;
    const double diffusion=c.volatility*std::sqrt(c.time_to_expiry), discount=std::exp(-c.rate*c.time_to_expiry);
    const std::uint64_t pairs=paths/2; double mean=0.0,m2=0.0;
    for(std::uint64_t i=0;i<pairs;++i) {
        const double z=normal(rng), s1=c.spot*std::exp(drift+diffusion*z), s2=c.spot*std::exp(drift-diffusion*z);
        const double x=discount*0.5*(payoff(c.type,s1,c.strike)+payoff(c.type,s2,c.strike));
        const double delta=x-mean; mean+=delta/static_cast<double>(i+1); m2+=delta*(x-mean);
    }
    const double se=std::sqrt(m2/(static_cast<double>(pairs-1)*static_cast<double>(pairs)));
    return {mean,se,mean-1.959963984540054*se,mean+1.959963984540054*se,paths};
}

std::optional<ImpliedVolResult> implied_volatility(Contract c,double market_price,double price_tol,double vol_tol,std::size_t max_iter) {
    validate(c);
    if(c.style!=ExerciseStyle::european) throw std::invalid_argument("implied volatility solver supports European exercise only");
    if(!std::isfinite(market_price)||!std::isfinite(price_tol)||!std::isfinite(vol_tol)||price_tol<=0||vol_tol<=0||max_iter==0)
        throw std::invalid_argument("invalid implied-volatility solver arguments");
    if(c.time_to_expiry==0.0 || market_price<0.0) return std::nullopt;
    const double lower=c.type==OptionType::call?std::max(c.spot*std::exp(-c.dividend_yield*c.time_to_expiry)-c.strike*std::exp(-c.rate*c.time_to_expiry),0.0):std::max(c.strike*std::exp(-c.rate*c.time_to_expiry)-c.spot*std::exp(-c.dividend_yield*c.time_to_expiry),0.0);
    const double upper=c.type==OptionType::call?c.spot*std::exp(-c.dividend_yield*c.time_to_expiry):c.strike*std::exp(-c.rate*c.time_to_expiry);
    if(market_price<lower-price_tol || market_price>=upper || market_price==lower) return std::nullopt;
    double lo=1e-8,hi=0.5; c.volatility=hi;
    while(black_scholes(c).price<market_price && hi<8.0) {hi*=2.0;c.volatility=hi;}
    if(black_scholes(c).price<market_price) return std::nullopt;
    double sigma=std::clamp(0.2,lo,hi);
    for(std::size_t i=1;i<=max_iter;++i) {
        c.volatility=sigma; const auto model=black_scholes(c); const double error=model.price-market_price;
        if(std::abs(error)<=price_tol || hi-lo<=vol_tol)
            return ImpliedVolResult{sigma,model.price,error,i,true,"safeguarded-newton"};
        if(error>0) hi=sigma; else lo=sigma;
        double next=sigma-error/model.vega;
        if(!std::isfinite(next)||next<=lo||next>=hi||model.vega<1e-14) next=0.5*(lo+hi);
        sigma=next;
    }
    return std::nullopt;
}
} // namespace volterra
