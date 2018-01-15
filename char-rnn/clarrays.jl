using CLArrays

gpudevs = CLArrays.devices(is_gpu)
hasCL = length(gpudevs) > 0

function useCL(Xs, Ys, ngpu=1000)
    hasCL || return Xs, Ys
    CLArrays.init(gpudevs[1])
    CXs = [[CLArray(Array{Float32}(m)) for m in x] for x in Xs[1:ngpu]]
    CYs = [[CLArray(Array{Float32}(m)) for m in y] for y in Ys[1:ngpu]]
    return CXs, CYs
end
    
