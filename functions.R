## Some self-defined functions

# function to standardize nurmerical covariates
standardize <- function(x){
  z <- (x - mean(x, na.rm = T)) / sd(x, na.rm = T)
  return( z)
}

# function to impute NA as mean
na.impute <- function(x){
  for (i in 1:length(x)){
    if (is.na(x[i])) x[i] = mean(x, na.rm = T)
  }
  return(x)
}

# function for method 1: Reweighing
reweighing <- function(data){
  # method 1: reweighting (possible to have 0, possible to be negative)
  fitC = survfit(Surv(time, (1-event))~1, data,  type=c("kaplan-meier"))
  # plot(fitC, mark.time = F, xlab = "Time", ylab = "Survival Probability")
  KM_c = summary(fitC,times=data$time,extend=TRUE) #KM_c is now ordered by time
  
  y.new = NULL
  for(i in 1:nrow(data)){
    if(data[i,]$event == 1){
      idx = which(KM_c$time == data[i,]$time)[1]
      interim = log(data[i,]$time)/KM_c$surv[idx] # the h() function is log here
      #interim = log(data[i,]$time)
    }else{
      #idx = which(KM_c$time == data[i,]$time)
      #interim = log(data[i,]$time)/KM_c$surv[idx]
      interim = 0
    }
    
    y.new = c(y.new, interim)
  }
  data = cbind(data, y.new.m1=y.new)
  return(data)
}

# function for method 2: Mean Imputation
mean.imputation <- function(data){
  # method 2: mean imputation (rare chance to have 0, possible to be negative)
  fit = survfit(Surv(time, event)~1, data,  type=c("kaplan-meier"))
  plot(fit, mark.time = F, xlab = "Time", ylab = "Survival Probability")
  KM = summary(fit,times=unique(data$time) ,extend=TRUE) #KM_c is now ordered by time
  
  fail.t = data[data$event==1,]$time
  sort.t = sort(fail.t)
  
  y.new = NULL 
  for(i in 1:nrow(data)){
    if(data[i,]$event == 1){
      interim = log(data[i,]$time)
    }else{
      idx = which(KM$time == data[i,]$time)#[1]
      tau.idx = which(sort.t > data[i,]$time)
      if(length(tau.idx)>0){
        log.cum = 0
        for(j in tau.idx){
          idx.j = which(KM$time == sort.t[j])
          if (j==1){
            log.cum = log.cum + log(sort.t[j])*(1 - KM$surv[idx.j])
          }else{
            idx.j1 = which(KM$time == sort.t[j-1]) # sort.t[j-1] becomes NA if j = 1 when tau.idx includes 1
            log.cum = log.cum + log(sort.t[j])*(KM$surv[idx.j1] - KM$surv[idx.j])
          }
          # print(paste(j, "," ,log.cum))
        }
      }else{
        log.cum = 0
      }
      
      interim = log.cum/KM$surv[idx] # the h() function is log here
      
    }
    
    y.new = c(y.new, interim)
  }
  
  y.new[is.na(y.new)] = 0
  data = cbind(data, y.new.m2=y.new)
  return(data)
}

# function for method 3: Deviance Residual
deviance.residual <- function(data){ 
  # method 3 deviance residual # \approx observed - expected (possible to be negative)
  null_obj = coxph(Surv(time,event) ~ 1, data = data, x = T, y = T) 
  new.y = residuals(null_obj, type = 'deviance')
  
  data = cbind(data, y.new.m3=new.y)
  return(data)
}

# function for feature extraction with customized NN
Surv_NN <- function(data, y.var, x.var, hidden.nodes = c(20), activations = c('relu'), output.activation = "relu",
                    batch.normalization = F, batch.size, epochs, verbose = 0, val.rate = 0.1, 
                    early.stop = F, early.patience = 10,
                    split.rate=0.8, split.folds=NULL, seed = 100){ 
  # Split training & testing sets
  n = nrow(data)
  if (is.null(split.folds)){
    set.seed(seed)
    train.no = sample(1:n, round(n*split.rate), replace = F)
  }else{
    train.no = split.folds
  }
  
  train.ind = vector("logical", length=n)
  train.ind[train.no] = 1
  train.ind[-train.no] = 0
  
  x_train = data[train.no, x.var] # remove id column
  y_train = data[train.no, y.var]
  x_test = data[-train.no, x.var] # remove id column
  y_test = data[-train.no, y.var]
  x_all = data[, x.var]
  
  #scaling
  x_scaled = apply(x_all, 2, function(x){(x-min(x))/(max(x)-min(x))})
  x_train_scaled = x_scaled[train.no,]
  x_test_scaled = x_scaled[-train.no,]
  ## ------------------------------------------------------
  k_clear_session()
  # Define ANN arthitecture
  hidden.no = length(hidden.nodes)
  model <- keras_model_sequential()%>% 
    layer_dense(units = hidden.nodes[1], activation=activations[1], input_shape = length(x.var))
  if (batch.normalization){model %>% layer_batch_normalization()}
  if (hidden.no > 1){for (i in 2:hidden.no){ 
    model%>%
      layer_dense(units = hidden.nodes[i], activation=activations[i])
    if (batch.normalization){model %>% layer_batch_normalization()}
  }}
  model%>%  
    layer_dense(units = length(y.var), activation=output.activation) 
  # Compile the model
  model %>% compile(loss="mse", optimizer=optimizer_adam())
  
  # Train the model
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = early.patience, mode = "auto")
  if (early.stop){
    model %>% fit(x = x_train_scaled, y = y_train, validation_split = val.rate,
                  batch_size = batch.size, epochs = epochs, verbose = verbose,
                  callbacks = list(early_stop))
  }else{
    history <- model %>% fit(
      x = x_train_scaled, y = y_train, validation_split = val.rate,
      batch_size = batch.size, epochs = epochs, verbose = verbose)
  }
  
  # Make prediction
  y_pred = model %>% predict(x_scaled)
  
  # Extract features for tarining/test sets
  model_feature <- keras_model_sequential()%>%  
    layer_dense(units = hidden.nodes[1], activation=activations[1], input_shape = length(x.var))
  if (hidden.no > 1){for (i in 2:hidden.no){
    if (batch.normalization){model %>% layer_batch_normalization()}
    model_feature %>%
      layer_dense(units = hidden.nodes[i], activation=activations[i])
  }}
  model_feature$set_weights(get_weights(model)[1:(2*(length(model$layers)-1))])#[1:(2*hidden.no)]
  model_feature %>% compile(loss="mse", optimizer=optimizer_adam())
  feature = model_feature %>% predict(x_scaled)
  # feature_train = model_feature %>% predict(x_train_scaled)
  # feature_test = model_feature %>% predict(x_test_scaled)
  
  data_pred = cbind(data, time.pred = y_pred, train.indicator = train.ind)
  data_feature = cbind(data, feature = feature, train.indicator = train.ind)
  # data_feature = cbind(data[,!names(data) %in% x.var], feature = feature, train.indicator = train.ind)
  
  return(list(data = data_pred, data_feature = data_feature))
}
