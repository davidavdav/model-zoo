# using StatsBase

function sample(x, w)
    w = w ./ sum(w)
    cw = cumsum(w)
    r = rand()
    i = 1
    while r > cw[i]
        i += 1
    end
    return x[i]
end


function sampleT(m, alphabet, len)
    Flux.reset!(m)
    buf = IOBuffer()
    c = rand(alphabet)
    write(buf, c)
    s = onehot(c, alphabet)
    for i = 1:len
        s = m(s)
        # don't just take the maximum, but use s as probabilities
        # and sample from it. Otherwise the text gets repetetive.
        c = sample(alphabet, s.data)
        write(buf, c)
        # then start from what has been done (sampled), not from what had been predicted.
        s = onehot(c, alphabet)
    end
    return String(take!(buf))
end
