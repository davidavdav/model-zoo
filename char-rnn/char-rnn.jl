using Flux
using Flux: onehot, argmax, chunk, batchseq, truncate!, throttle, crossentropy
using Base.Iterators: partition

using CLArrays

gpudevs = CLArrays.devices(is_gpu)
useCL = length(gpudevs) > 0
if useCL
    CLArrays.init(gpudevs[1])
end

cd(@__DIR__)

isfile("input.txt") ||
  download("http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

text = collect(readstring("input.txt"))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 50
nbatch = 50

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

m = Chain(
  LSTM(N, 128),
  LSTM(128, 256),
  LSTM(256, 128),
  Dense(128, N),
  softmax)

loss(xs, ys) = sum(crossentropy.(m.(xs), ys))

opt = ADAM(params(m), Float32(0.01))

evalcb = () -> @show loss(Xs[5], Ys[5])

trainstep(Xs, Ys) = Flux.train!(loss, zip(Xs, Ys), opt, cb = [() -> truncate!(m), throttle(evalcb, 10)])

if useCL
    ngpu = 100
    CXs = [[CLArray(Array{Float32}(m)) for m in x] for x in Xs[1:ngpu]]
    CYs = [[CLArray(Array{Float32}(m)) for m in y] for y in Ys[1:ngpu]]
end

# Sampling

function orig_sample(m, alphabet, len)
  Flux.reset!(m)
  buf = IOBuffer()
  s = onehot(rand(alphabet), alphabet)
  for i = 1:len
    write(buf, argmax(s, alphabet))
    s = m(s)
  end
  return String(take!(buf))
end
