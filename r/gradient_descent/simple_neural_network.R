# This module implements simple Artificial Neural Network 
# The ANN consists of: 
#   - input layer of dimensions (n.x,1), where n.x is number of input values
#   - n.h hidden layers each consisting of n.hl number of nodes (neurons),
#     where l is layer number
#   - single value output
# The Network is optimised by Stochastic Gradient Descent 
# Loss (error) function used is Binary Cross Entropy:
# L(y.hat,y) = -(y.i*log(y.hat.i) + (1-y.i)*log(1-y.hat.i)), where:
# y.i - true value for i-th sample, y.hat.i - ANN prediction for i-th sample
# Gradient Descent minimizes cost function:
# I(w) = 1/m * sum(L(y.hat,y)), where:
# w - network's weights and biases 


# ---------- HELPER FUNCTIONS ----------

# Constructs matrices for weights and biases, initializes them with random values
get.weights.and.biases <- function(input.length, hidden.lengths, output.length) {
  W <- list()
  B <- list()
  n <- input.length
  layer.lengths <- c(hidden.lengths, output.length)
  for (i in 1:(length(layer.lengths))) {
    l <- layer.lengths[i]
    Wl <- array(data=rnorm(n*l), dim=c(n,l))
    bl <- array(data=rnorm(l), dim=c(1,l))
    W <- c(W, list(Wl))
    B <- c(B, list(bl))
    n <- l
  }
  result <- list(weights=W, biases=B)
}

# Broadcasts bias and adds it to array
add.bias <- function(mat, bias) {
  dims <- dim(mat)
  bias <- t(array(data=bias,dim=c(dims[2], dims[1])))
  result <- mat + bias
}

split.to.minibatches <- function(minibatch, X, Y) {
  # shuffle the data
  XY <- cbind(X,Y)
  data.length <- dim(XY)[1]
  XY <- XY[sample(1:data.length),]
  # split data
  XYout <- list()
  start <- 1
  # If minibatches < 1 then all dataset is in one big batch
  if(minibatch == 0){
    minibatch <- data.length
  }
  stop <- minibatch
  while(stop <= data.length) {
    XYout <-c(XYout, list(XY[start:stop,,drop=FALSE]))
    start <- stop + 1
    stop <- stop + minibatch
  }
  if (start <= data.length){
    XYout <-c(XYout, list(XY[start:data.length,,drop=FALSE]))
  }
  result <- XYout
}

# ---------- MATH FUNCTIONS ----------
# Sigmoid activation function
sigmoid <- function(x) {
  result <- 1/(1 + exp(-x))
}

# Derivative of sigmoid activation function
dsigmoid <- function(sigmoid) {
  result <- sigmoid*(1 - sigmoid)
}

# Derivative of loss function L with regards to prediction y.hat
dLdAout <- function(Aout, Y) {
  result <- (1-Y)/(1-Aout) - Y/Aout
}

# Delta for last layer
output.delta <- function(A, Y) {
  l <- length(A)
  Aout <- A[[l]]
  result <- dLdAout(Aout, Y) * dsigmoid(Aout)
}

layer.delta <- function(l, delta.prev, W, A) {
  result <- (delta.prev %*% t(W[[l]])) * dsigmoid(A[[l]])
}

compute.cost <- function(Aout, Y) {
  m <- prod(dim(Y))
  loss <- Y * log10(Aout) + (1-Y)*log10(1-Aout)
  cost <- - sum(loss)/m
  result <- cost
}

# ---------- GRADIENT DESCENT FUNCTIONS ----------
forward.propagation <- function(X, W, B) {
  # Declare cache of acivations as list
  Acache <- list()
  Acache <- c(Acache, list(X))
  # Length of list of weight matrices is a number of hidden layers + 1 for output layer
  layers = length(W)
  # Forward propagation loop - iterate though layers and calculate activation values of 
  # each node
  for (l in 1:layers) {
    z <- add.bias(Acache[[l]] %*% W[[l]], B[[l]]) 
    a <- sigmoid(z)
    Acache <- c(Acache, list(a))
  }
  # Return cache list of activations
  result <- Acache
}

back.propagation <- function(A, W, B, Y, alpha) {
  layers = length(W)
  delta.out <- output.delta(A, Y)
  delta.prev <- delta.out
  
  for(l in (layers):1) {
    dW <- alpha * t(A[[l]]) %*% delta.prev
    db <- alpha * delta.prev
    delta.l <- layer.delta(l,delta.prev,W,A)
    delta.prev <- delta.l
    W[[l]] <- W[[l]] - dW
    B[[l]] <- add.bias(-db,B[[l]]) 
  }
  result <- W
}

# ---------- NEURAL NETWORK MAIN FUNCTIONS ----------
train <- function(X, Y, WB, epochs, learning.rate, minibatch = 0) {
  # unpack model parameters
  W = WB$weights
  B = WB$biases
  # split to minibatches
  XY <- split.to.minibatches(minibatch, X, Y)
  pb <- txtProgressBar(min = 1, max = 100, style = 3)
  # initialize array for cost function recording
  cost.array <- array(data=0, dim=c(epochs,1))
  for(e in 1:epochs) {
    setTxtProgressBar(pb, round((e/epochs)*100,0))
    getTxtProgressBar(pb)
    for(b in 1:length(XY)){
      # unpack X, Y
      X <- XY[[b]][,1:(dim(X)[2]), drop = FALSE]
      Y <- XY[[b]][,(dim(X)[2]+1):(dim(X)[2]+dim(Y)[2]), drop = FALSE]
      A <- forward.propagation(X,W,B)
      W <- back.propagation(A,W,B,Y,learning.rate)
    }
    cst <- compute.cost(tail(A, n = 1)[[1]],Y)
    cost.array[e] <- cst
  }
  close(pb)
  # Pack everything and return
  trained_model <- list(weights = W, biases = B, cost = cost.array)
  result <- trained_model
}

# Run forward propagation to get predictions
predict <- function(X,W,B) {
  # unpack model parameters
  A <- forward.propagation(X,W,B)
  prediction <- tail(A,n=1)[[1]]
  result <- prediction
}

# --------- DEBUG ----------

# For debug and testing - creates array with random data
get.random.matrix <- function(samples, length, binary = FALSE) {
  random.data <- runif(length*samples,0,1)
  if(binary == TRUE) {
    random.data <- round(random.data,0)
  }
  v <- array(data=random.data,dim=c(samples,length))
}

test <- function() {
  X <- get.random.matrix(1000,2)
  Y <- get.random.matrix(1000,1,TRUE)
  WB <- get.weights.and.biases(2,3,1)
  train(X,Y,WB,1000,0.05,minibatch = 8,10)
}
