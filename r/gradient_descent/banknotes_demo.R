source("simple_neural_network.R")

accuracy <- function(Y.out, Y, treshold = 0.5) {
  Y.pred <- as.integer(Y.out >= treshold)
  result <- sum(as.integer(Y.pred == Y))/length(Y)
}

banknotes_demo <- function() {
  # Read in banknotes dataset
  banknotes = read.table("data_banknote_authentication.txt", sep=",")
  # Convert list to array
  bn.array <- array(as.numeric(unlist(banknotes)), dim=dim(banknotes))
  # Split dataset to test and train
  sample.size = floor(0.75*nrow(bn.array))
  train.indices = sample(seq_len(nrow(bn.array)), size = sample.size)
  train.set = bn.array[train.indices,]
  val.set = bn.array[-train.indices,]
  # Unpack X and Y from data array
  X.train <- train.set[,1:dim(train.set)[2]-1]
  Y.train <- train.set[,dim(train.set)[2], drop = FALSE]
  X.val <- val.set[,1:dim(val.set)[2]-1]
  Y.val <- val.set[,dim(val.set)[2], drop = FALSE]
  # Initialize NN structure
  WB <- get.weights.and.biases(4,5,1)
  epochs <- 5
  print("Training model:")
  model <- train(X.train, Y.train, WB, epochs, 0.05, minibatch = 128)
  plot(c(1:epochs),model$cost)
  
  # Accuracy
  Y.out <- predict(X.train, model$weights, model$biases)
  print("Accuracy on training set:")
  print(accuracy(Y.out, Y.train))
  
  Y.out <- predict(X.val, model$weights, model$biases)
  print("Accuracy on validation set:")
  print(accuracy(Y.out, Y.val))
}