function neuron(input, weight)
  return input * weight
end

function calculate_error(prediction, output)
  return (prediction - output) ^ 2
end

function calculate_gradient(prediction, output)
  return 2 * prediction - 2 * output
end

input = 0.5
weights = { 0.85, 0.56 }
output = 0.3
epochs = 500
learning_rate = 0.01

for epoch = 1, epochs do
  print('Epoch', epoch)
  hidden_prediction = neuron(input, weights[1])
  prediction = neuron(hidden_prediction, weights[2])

  hidden_gradient = calculate_gradient(hidden_prediction, prediction)
  weights[1] = weights[1] - hidden_gradient * learning_rate

  gradient = calculate_gradient(prediction, output)
  weights[2] = weights[2] - gradient * learning_rate

  print('Prediction', prediction)
  print('Error', calculate_error(prediction, output), '\n')
end
