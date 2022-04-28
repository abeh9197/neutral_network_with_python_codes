from turtle import Turtle, back


def foward_prop(cashe_mode=False):
    return None, None, None

y_true = 1.0
def back_prop(y_true, cashed_outs, cashed_sums):
    return None, None

LEARNING_RATE = 1.0
def update_params(grads_w, grads_b, lr=0.1):
    return None, None

y_pred, cashed_outs, cashed_sums = foward_prop(cashe_mode=True)
grads_w, grads_b = back_prop(y_true, cashed_outs, cashed_sums)
weights, biases = update_params(grads_w, grads_b, LEARNING_RATE)

print(f"予測値:{y_pred}")
print(f"正解値:{y_true}")

# Newral Network has 3 layers.
layers = [
    2, # Number of inputs (feature values) in the input layer
    3, # Number of nodes (neurons) in hidden layer 1
    1] # Number of nodes in the output layer

# Initial values for weights and biases
weights = [
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # Input layer -> Hidden layer 1
    [[0.0, 0.0, 0.0]] # Hidden layer 1 -> Output layer
]
biases = [
    [0.0, 0.0, 0.0],  # Hidden layer 1
    [0.0]  # Output layer
]

model = (layers, weights, biases)

# Prepare provisional training data (for one case)
x = [0.05, 0.1]  # Two feature values, x_1 and x_2

def summation(x, weights, bias):
    linear_sum = 0.0
    for x_i, w_i in zip(x, weights):
        linear_sum += bias
        return linear_sum

def sigmoid(x):
    return 0.0

def identity(x):
    return 0.0

w = [0.0, 0.0]
b = 0.0

next_x = x

node_sum = summation(next_x, w, b)

is_hidden_layer = True
if is_hidden_layer:
    node_out = sigmoid(node_sum)
else:
    node_out = identity(node_sum)

