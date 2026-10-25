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

MonteCarloResult american_monte_carlo_price(const Contract& c,std::uint64_t paths,
                                             std::size_t steps,std::uint64_t seed) {
    validate(c);
    if(c.style!=ExerciseStyle::american) throw std::invalid_argument("Longstaff-Schwartz requires American exercise style");
    if(paths<4 || steps==0) throw std::invalid_argument("American Monte Carlo requires at least four paths and one time step");
    if(paths%2!=0) --paths;
    if(c.time_to_expiry==0.0) {
        const double p=payoff(c.type,c.spot,c.strike);
        return {p,0.0,p,p,paths};
    }
    if(c.volatility==0.0) {
        const double p=binomial_price(c,steps);
        return {p,0.0,p,p,paths};
    }
    const std::size_t path_count=static_cast<std::size_t>(paths), columns=steps+1;
    if(columns>std::numeric_limits<std::size_t>::max()/path_count)
        throw std::length_error("American Monte Carlo path matrix is too large");
    const std::size_t pairs=path_count/2;
    std::vector<double> spots(path_count*columns);
    const double dt=c.time_to_expiry/static_cast<double>(steps);
    const double drift=(c.rate-c.dividend_yield-.5*c.volatility*c.volatility)*dt;
    const double diffusion=c.volatility*std::sqrt(dt);
    std::mt19937_64 rng(seed); std::normal_distribution<double> normal;
    for(std::size_t p=0;p<pairs;++p) {
        double up_spot=c.spot, down_spot=c.spot;
        spots[p*columns]=c.spot; spots[(pairs+p)*columns]=c.spot;
        for(std::size_t t=1;t<columns;++t) {
            const double z=normal(rng);
            up_spot*=std::exp(drift+diffusion*z);
            down_spot*=std::exp(drift-diffusion*z);
            if(!std::isfinite(up_spot)||!std::isfinite(down_spot))
                throw std::overflow_error("simulated option path overflowed; check volatility and expiry inputs");
            spots[p*columns+t]=up_spot;
            spots[(pairs+p)*columns+t]=down_spot;
        }
    }

    std::vector<double> cashflow(path_count);
    std::vector<std::size_t> exercise_step(path_count,steps);
    for(std::size_t p=0;p<path_count;++p)
        cashflow[p]=payoff(c.type,spots[p*columns+steps],c.strike);

    for(std::size_t step=steps;step-->1;) {
        double max_abs_x=0.0; std::size_t itm_count=0;
        for(std::size_t p=0;p<path_count;++p) {
            const double s=spots[p*columns+step];
            if(payoff(c.type,s,c.strike)>0.0) {
                max_abs_x=std::max(max_abs_x,std::abs(std::log(s/c.strike)));
                ++itm_count;
            }
        }
        if(itm_count<4) continue;
        const double scale=std::max(max_abs_x,1e-12);
        double matrix[3][4]{};
        for(std::size_t p=0;p<path_count;++p) {
            const double s=spots[p*columns+step], immediate=payoff(c.type,s,c.strike);
            if(immediate<=0.0) continue;
            const double x=std::log(s/c.strike)/scale, basis[3]={1.0,x,x*x};
            const double y=cashflow[p]*std::exp(-c.rate*static_cast<double>(exercise_step[p]-step)*dt);
            for(int row=0;row<3;++row) {
                for(int col=0;col<3;++col) matrix[row][col]+=basis[row]*basis[col];
                matrix[row][3]+=basis[row]*y;
            }
        }
        bool fit_ok=true;
        for(int col=0;col<3;++col) {
            int pivot=col;
            for(int row=col+1;row<3;++row)
                if(std::abs(matrix[row][col])>std::abs(matrix[pivot][col])) pivot=row;
            if(std::abs(matrix[pivot][col])<1e-12) {fit_ok=false;break;}
            if(pivot!=col) for(int j=col;j<4;++j) std::swap(matrix[pivot][j],matrix[col][j]);
            const double divisor=matrix[col][col];
            for(int j=col;j<4;++j) matrix[col][j]/=divisor;
            for(int row=0;row<3;++row) if(row!=col) {
                const double factor=matrix[row][col];
                for(int j=col;j<4;++j) matrix[row][j]-=factor*matrix[col][j];
            }
        }
        if(!fit_ok) continue;
        const double beta[3]={matrix[0][3],matrix[1][3],matrix[2][3]};
        for(std::size_t p=0;p<path_count;++p) {
            const double s=spots[p*columns+step], immediate=payoff(c.type,s,c.strike);
            if(immediate<=0.0) continue;
            const double x=std::log(s/c.strike)/scale;
            const double continuation=beta[0]+beta[1]*x+beta[2]*x*x;
            if(immediate>continuation) {cashflow[p]=immediate;exercise_step[p]=step;}
        }
    }

    std::vector<double> pair_values(pairs);
    for(std::size_t p=0;p<pairs;++p) {
        const double first=cashflow[p]*std::exp(-c.rate*static_cast<double>(exercise_step[p])*dt);
        const double second=cashflow[pairs+p]*std::exp(-c.rate*static_cast<double>(exercise_step[pairs+p])*dt);
        pair_values[p]=.5*(first+second);
    }
    double mean=0.0,m2=0.0;
    for(std::size_t p=0;p<pairs;++p) {
        const double delta=pair_values[p]-mean;
        mean+=delta/static_cast<double>(p+1);
        m2+=delta*(pair_values[p]-mean);
    }
    const double intrinsic=payoff(c.type,c.spot,c.strike);
    const bool exercise_now=intrinsic>mean;
    if(exercise_now) mean=intrinsic;
    const double se=exercise_now?0.0:std::sqrt(m2/(static_cast<double>(pairs-1)*static_cast<double>(pairs)));
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
