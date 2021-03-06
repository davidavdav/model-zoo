using Flux, MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated

# Classify MNIST digits with a simple multi-layer-perceptron

x, y = traindata()
y = onehotbatch(y, 0:9)

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax)

# using CuArrays
# x, y = cu(x), cu(y)
# m = mapleaves(cu, m)

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

dataset = repeated((x, y), 200)
evalcb = () -> @show(loss(x, y))
opt = ADAM(params(m))

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 5))

accuracy(x, y)

# Test set accuracy
tx, ty = testdata()
ty = onehotbatch(ty, 0:9)
accuracy(tx, ty)
