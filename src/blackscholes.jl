# Multi-threading in Julia
# Multi-threading will be native to Julia in the next release (v0.5.x). It is currently under active development. 
# This tutorial gives you a quick taste of how to use multi-threading in Julia. 

# First load the threading code that's in the Threads module.

using Base.Threads

@inline function cndf2(in::Array{Float64,1})
    out = 0.5 .+ 0.5 .* erf(0.707106781 .* in)
    return out
end

function blackscholes_serial(sptprice::Array{Float64,1},
                           strike::Array{Float64,1},
                           rate::Array{Float64,1},
                           volatility::Array{Float64,1},
                           time::Array{Float64,1})
    logterm = log10(sptprice ./ strike)
    powterm = .5 .* volatility .* volatility
    den = volatility .* sqrt(time)
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den
    d2 = d1 .- den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike .* exp(- rate .* time)
    c1 = futureValue .* NofXd2
    call = sptprice .* NofXd1 .- c1
    put  = call .- futureValue .+ sptprice
end

# This next parallel rewrite does two simple things:
# It devectorizes the entire loop. We're now looping through arrays instead of dealing with them in blocks like we did previously. This is devectorized code while the above serial block is vectorized code. 
# Affixes the `@threads` macro in front of the `for` to tell Julia that this is a multi-threaded block.

function blackscholes_parallel(sptprice::Vector{Float64}, strike::Vector{Float64}, rate::Vector{Float64}, volatility::Vector{Float64}, time::Vector{Float64})
    sqt = sqrt(time)
    put = similar(strike)
    @threads for i = 1:size(sptprice, 1)
        logterm = log10(sptprice[i] / strike[i])
        powterm = 0.5 * volatility[i] * volatility[i]
        den = volatility[i] * sqt[i]
        d1 = (((rate[i] + powterm) * time[i]) + logterm) / den
        d2 = d1 - den
        NofXd1 = 0.5 + 0.5 * erf(0.707106781 * d1)
        NofXd2 = 0.5 + 0.5 * erf(0.707106781 * d2)
        futureValue = strike[i] * exp(-rate[i] * time[i])
        c1 = futureValue * NofXd2
        call = sptprice[i] * NofXd1 - c1
        put[i] = call - futureValue + sptprice[i]
    end
    put
end


function run(iterations)
    sptprice   = Float64[ 42.0 for i = 1:iterations ]
    initStrike = Float64[ 40.0 + (i / iterations) for i = 1:iterations ]
    rate       = Float64[ 0.5 for i = 1:iterations ]
    volatility = Float64[ 0.2 for i = 1:iterations ]
    time       = Float64[ 0.5 for i = 1:iterations ]

    tic()
    put1 = blackscholes_serial(sptprice, initStrike, rate, volatility, time)
    t1 = toq()
    println("Serial checksum: ", sum(put1))
    tic()
    put2 = blackscholes_parallel(sptprice, initStrike, rate, volatility, time)
    t2 = toq()
    println("Parallel checksum: ", sum(put2))
    return t1, t2
end

function driver()
    srand(0)
    tic()
    iterations = 10^6
    blackscholes_serial(Float64[], Float64[], Float64[], Float64[], Float64[])
    blackscholes_parallel(Float64[], Float64[], Float64[], Float64[], Float64[])
    println("SELFPRIMED ", toq())
    tserial, tparallel = run(iterations)
    println("Time taken for serial = $tserial")
    println("Time taken for parallel = $tparallel")
    println("Serial rate = ", iterations / tserial, " opts/sec")
    println("Parallel rate = ", iterations / tparallel, " opts/sec")
end
driver()
