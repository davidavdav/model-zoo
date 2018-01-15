using CuArrays

function useCuda(Xs, Ys, ngpu=1000)
    CXs = [[CuArray(Array{Float32}(m)) for m in x] for x in Xs[1:ngpu]]
    CYs = [[CuArray(Array{Float32}(m)) for m in y] for y in Ys[1:ngpu]]
    return CXs, CYs
end


    
