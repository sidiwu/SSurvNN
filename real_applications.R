library(survival)
library(fastDummies)
library(tensorflow)
library(keras)
library(caret)
library(pbapply)
library(tidyverse)
setwd("~")
source("functions.R")

############################
##### ---- metabric--- #####
############################
##--- Import data
metabric <- read.csv("datasets/real_datasets/metabric.csv")
str(metabric)
anyNA(metabric)

## k-fold CV
nfolds = 5
set.seed(823)
metabric.folds = createFolds(1:nrow(metabric[metabric$duration!=0,]), k = nfolds, list = T, returnTrain = F)
metabric.nfolds.cindex.nn.cox = metabric.nfolds.cindex.nn.cox.np = data.frame(matrix(NA, ncol = 4, nrow = nfolds))
names(metabric.nfolds.cindex.nn.cox) = names(metabric.nfolds.cindex.nn.cox.np) = c("Classic_Cox", "RW", "MI", "DR")

for (i in 1:nfolds){
  # Categorical covariates
  # Binary variables: x4, x5, x6, x7
  metabric.data = metabric.data.classic = metabric
  metabric.data.classic = metabric.data.classic[metabric.data.classic$duration!=0,]
  
  # Remove subjects with time 0
  metabric.data.drop = metabric.data[metabric.data$duration!=0,]
  
  split.folds = c(1:nrow(metabric.data.drop))[-metabric.folds[[i]]]
  # Numerical covariates
  num_cov_metabric = c("x0", "x1", "x2", "x3", "x8")
  metabric.data[split.folds,num_cov_metabric] = apply(metabric.data[split.folds,num_cov_metabric], 2, standardize)
  metabric.data[-split.folds,num_cov_metabric] = apply(metabric.data[-split.folds,num_cov_metabric], 2, standardize)
  
  ## ----  Classic Cox ------
  metabric.cox.temp <- coxph(Surv(duration, event) ~ ., data = metabric.data.classic[split.folds, ])
  metabric.nfolds.cindex.nn.cox$Classic_Cox[i] = metabric.nfolds.cindex.nn.cox.np$Classic_Cox[i] = 
    concordance(metabric.cox.temp, newdata = metabric.data.classic[-split.folds, ])$concordance
  
  ##--- Transformation
  metabric.y.m1 = metabric.y.m2 = metabric.y.m3 = vector("logical", length = nrow(metabric.data.drop))
  # method 1: reweighting (possible to have 0, possible to be negative)
  metabric.data.m1 = reweighing(data = data.frame(time = metabric.data.drop[split.folds,]$duration, event = metabric.data.drop[split.folds,]$event))
  metabric.y.m1[split.folds] = metabric.data.m1$y.new.m1
  # method 2: mean imputation (rare chance to have 0, possible to be negative)
  metabric.data.m2 = mean.imputation(data = data.frame(time = metabric.data.drop[split.folds,]$duration, event = metabric.data.drop[split.folds,]$event))
  metabric.y.m2[split.folds] = metabric.data.m2$y.new.m2
  # method 3: deviance residual # \approx observed - expected (possible to be negative)
  metabric.data.m3 = deviance.residual(data = data.frame(time = metabric.data.drop[split.folds,]$duration, event = metabric.data.drop[split.folds,]$event))
  metabric.y.m3[split.folds] = metabric.data.m3$y.new.m3
  
  ##--- Feature extraction with NN
  metabric.x.var = names(metabric.data.drop)[1:9]
  
  metabric.n_feature = 15
  metabric.f.m1.temp = Surv_NN(data = cbind(metabric.data.drop, y.m1=metabric.y.m1), y.var = "y.m1", x.var = metabric.x.var, 
                               hidden.nodes = c(30, metabric.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = 'linear', 
                               split.folds = split.folds, batch.size = 32, epochs = 100)
  
  metabric.f.m2.temp = Surv_NN(data = cbind(metabric.data.drop, y.m2=metabric.y.m2), y.var= "y.m2", x.var = metabric.x.var,
                               hidden.nodes =c(30, metabric.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = 'linear',
                               split.folds = split.folds,batch.size = 32, epochs = 100)
  
  metabric.f.m3.temp = Surv_NN(data = cbind(metabric.data.drop, y.m3=metabric.y.m3), y.var= "y.m3", x.var = metabric.x.var, 
                               hidden.nodes = c(30, metabric.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = 'linear', 
                               split.folds = split.folds, batch.size = 32, epochs = 100) 
  
  ## ----  NN-Cox ------
  form.cox = "Surv(duration, event) ~ feature.1"
  for (j in 2: metabric.n_feature){form.cox = paste0(form.cox, "+feature.", j)}
  metabric.surv_form.cox = as.formula(form.cox)
  # RW-NN-Cox
  metabric.cox.m1.test.temp <- coxph(metabric.surv_form.cox, data = metabric.f.m1.temp$data_feature[split.folds,])
  metabric.nfolds.cindex.nn.cox$RW[i] = concordance(metabric.cox.m1.test.temp, newdata = metabric.f.m1.temp$data_feature[-split.folds,])$concordance
  # MI-NN-Cox
  metabric.cox.m2.test.temp <- coxph(metabric.surv_form.cox, data = metabric.f.m2.temp$data_feature[split.folds,])
  metabric.nfolds.cindex.nn.cox$MI[i] = concordance(metabric.cox.m2.test.temp, newdata = metabric.f.m2.temp$data_feature[-split.folds,])$concordance
  # DR-NN-Cox
  metabric.cox.m3.test.temp <- coxph(metabric.surv_form.cox, data = metabric.f.m3.temp$data_feature[split.folds,])
  metabric.nfolds.cindex.nn.cox$DR[i] = concordance(metabric.cox.m3.test.temp, newdata = metabric.f.m3.temp$data_feature[-split.folds,])$concordance
  
  ## ----  NN-Cox(NP) ------
  # Apply time splitting
  metabric.time.interval = 100
  metabric.cox.var = c("duration", "event", paste0("feature.", c(1:metabric.n_feature)))
  metabric.f.m1.cox.temp = survSplit(Surv(duration, event)~., data = metabric.f.m1.temp$data_feature[, metabric.cox.var],
                                     cut = seq(0, max(metabric.f.m1.temp$data_feature$duration), metabric.time.interval)[-1], 
                                     episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  metabric.f.m2.cox.temp = survSplit(Surv(duration, event)~., data = metabric.f.m2.temp$data_feature[, metabric.cox.var],
                                     cut = seq(0, max(metabric.f.m1.temp$data_feature$duration), metabric.time.interval)[-1], 
                                     episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  metabric.f.m3.cox.temp = survSplit(Surv(duration, event)~., data = metabric.f.m3.temp$data_feature[, metabric.cox.var],
                                     cut = seq(0, max(metabric.f.m1.temp$data_feature$duration), metabric.time.interval)[-1], 
                                     episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  
  # Set up formula
  form = "Surv(tstart, tstop, event) ~ feature.1"
  for (f in 2: metabric.n_feature){form = paste0(form, "+feature.", f)}
  set.seed(82); jj = sample(1:metabric.n_feature, ceiling(metabric.n_feature/2))

  for (j in jj){
    form = paste0(form, "+t.feature.",j)
    metabric.f.m1.cox.temp[, paste0("t.feature.", j)] = metabric.f.m1.cox.temp$tstart*metabric.f.m1.cox.temp[,paste0("feature.", j)]
    metabric.f.m2.cox.temp[, paste0("t.feature.", j)] = metabric.f.m2.cox.temp$tstart*metabric.f.m2.cox.temp[,paste0("feature.", j)]
    metabric.f.m3.cox.temp[, paste0("t.feature.", j)] = metabric.f.m3.cox.temp$tstart*metabric.f.m3.cox.temp[,paste0("feature.", j)]
  }
  metabric.surv_form = as.formula(form)
  
  # RW-NN-Cox(NP)
  metabric.cox.m1.test.temp.np <- coxph(metabric.surv_form, data = metabric.f.m1.cox.temp[metabric.f.m1.cox.temp$id %in% split.folds,])
  metabric.nfolds.cindex.nn.cox.np$RW[i] = concordance(metabric.cox.m1.test.temp.np, newdata = metabric.f.m1.cox.temp[!metabric.f.m1.cox.temp$id %in% split.folds,])$concordance
  # MI-NN-Cox(NP)
  metabric.cox.m2.test.temp.np <- coxph(metabric.surv_form, data = metabric.f.m2.cox.temp[metabric.f.m2.cox.temp$id %in% split.folds,])
  metabric.nfolds.cindex.nn.cox.np$MI[i] = concordance(metabric.cox.m2.test.temp.np, newdata = metabric.f.m2.cox.temp[!metabric.f.m2.cox.temp$id %in% split.folds,])$concordance
  # DR-NN-Cox(NP)
  metabric.cox.m3.test.temp.np <- coxph(metabric.surv_form, data = metabric.f.m3.cox.temp[metabric.f.m3.cox.temp$id %in% split.folds,])
  metabric.nfolds.cindex.nn.cox.np$DR[i] = concordance(metabric.cox.m3.test.temp.np, newdata = metabric.f.m3.cox.temp[!metabric.f.m3.cox.temp$id %in% split.folds,])$concordance
  
  print(paste("Fold", i, "is done!"))
}

colMeans(metabric.nfolds.cindex.nn.cox)
colMeans(metabric.nfolds.cindex.nn.cox.np)

############################
##### ---- gbsg --- #####
############################
##--- Import data
gbsg<- read.csv("datasets/real_datasets/gbsg.csv")
str(gbsg)
anyNA(gbsg)

## k-fold CV
nfolds = 5
set.seed(199)
gbsg.folds = createFolds(1:nrow(gbsg[gbsg$duration!=0,]), k = nfolds, list = T, returnTrain = F)
gbsg.nfolds.cindex.nn.cox = gbsg.nfolds.cindex.nn.cox.np = data.frame(matrix(NA, ncol = 4, nrow = nfolds))
names(gbsg.nfolds.cindex.nn.cox) = names(gbsg.nfolds.cindex.nn.cox.np) = c("Classic_Cox", "RW", "MI", "DR")

for (i in 1:nfolds){
  # Categorical covariates
  # Binary variables: x0, x2
  # Multi-level variables": x1(3)
  cat_cov_gbsg = c("x1")
  gbsg.data = gbsg.data.classic = dummy_cols(gbsg, select_columns = cat_cov_gbsg)
  gbsg.data.classic = gbsg.data.classic[gbsg.data.classic$duration!=0,]
  
  # Remove subjects with time 0
  gbsg.data.drop = gbsg.data[gbsg.data$duration!=0,]
  
  split.folds = c(1:nrow(gbsg.data.drop))[-gbsg.folds[[i]]]
  
  # Numerical covariates
  num_cov_gbsg = c("x3", "x4", "x5", "x6")
  gbsg.data[split.folds,num_cov_gbsg] = apply(gbsg.data[split.folds,num_cov_gbsg], 2, standardize)
  gbsg.data[-split.folds,num_cov_gbsg] = apply(gbsg.data[-split.folds,num_cov_gbsg], 2, standardize)
  
  ## ----  Classic Cox ------
  gbsg.cox.temp <- coxph(Surv(duration, event) ~ ., 
                         data = gbsg.data.classic[split.folds,!names(gbsg.data.classic) %in% cat_cov_gbsg])
  gbsg.nfolds.cindex.nn.cox$Classic_Cox[i] = gbsg.nfolds.cindex.nn.cox.np$Classic_Cox[i] = 
    concordance(gbsg.cox.temp, newdata = gbsg.data.classic[-split.folds,!names(gbsg.data.classic) %in% cat_cov_gbsg])$concordance
  
  ##--- Transformation
  gbsg.y.m1 = gbsg.y.m2 = gbsg.y.m3 = vector("logical", length = nrow(gbsg.data.drop))
  # method 1: reweighting (possible to have 0, possible to be negative)
  gbsg.data.m1 = reweighing(data = data.frame(time = gbsg.data.drop[split.folds,]$duration, event = gbsg.data.drop[split.folds,]$event))
  gbsg.y.m1[split.folds] = gbsg.data.m1$y.new.m1
  # method 2: mean imputation (rare chance to have 0, possible to be negative)
  gbsg.data.m2 = mean.imputation(data = data.frame(time = gbsg.data.drop[split.folds,]$duration, event = gbsg.data.drop[split.folds,]$event))
  gbsg.y.m2[split.folds] = gbsg.data.m2$y.new.m2
  # method 3: deviance residual # \approx observed - expected (possible to be negative)
  gbsg.data.m3 = deviance.residual(data = data.frame(time = gbsg.data.drop[split.folds,]$duration, event = gbsg.data.drop[split.folds,]$event))
  gbsg.y.m3[split.folds] = gbsg.data.m3$y.new.m3
  
  ##--- Feature extraction with NN
  gbsg.x.var = names(gbsg.data.drop)[1:7]
  
  gbsg.n_feature = 15
  gbsg.f.m1.temp = Surv_NN(data = cbind(gbsg.data.drop, y.m1=gbsg.y.m1), y.var = "y.m1", x.var = gbsg.x.var, 
                           hidden.nodes = c(30, gbsg.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear", 
                           split.folds = split.folds,batch.size = 24, epochs = 50)
  
  gbsg.f.m2.temp = Surv_NN(data = cbind(gbsg.data.drop, y.m2=gbsg.y.m2), y.var= "y.m2", x.var = gbsg.x.var,
                           hidden.nodes = c(30, gbsg.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear",
                           split.folds = split.folds,batch.size =24, epochs = 50)
  
  gbsg.f.m3.temp = Surv_NN(data = cbind(gbsg.data.drop, y.m3=gbsg.y.m3), y.var= "y.m3", x.var = gbsg.x.var, 
                           hidden.nodes = c(30, gbsg.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear", 
                           split.folds = split.folds, batch.size = 24, epochs = 50) 
  
  ## ----  NN-Cox ------
  form.cox = "Surv(duration, event) ~ feature.1"
  for (j in 2: gbsg.n_feature){form.cox = paste0(form.cox, "+feature.", j)}
  gbsg.surv_form.cox = as.formula(form.cox)
  
  # RW-NN-Cox
  gbsg.cox.m1.test.temp <- coxph(gbsg.surv_form.cox, data = gbsg.f.m1.temp$data_feature[split.folds,])
  gbsg.nfolds.cindex.nn.cox$RW[i] = concordance(gbsg.cox.m1.test.temp, newdata = gbsg.f.m1.temp$data_feature[-split.folds,])$concordance
  # MI-NN-Cox
  gbsg.cox.m2.test.temp <- coxph(gbsg.surv_form.cox, data = gbsg.f.m2.temp$data_feature[split.folds,])
  gbsg.nfolds.cindex.nn.cox$MI[i] = concordance(gbsg.cox.m2.test.temp, newdata = gbsg.f.m2.temp$data_feature[-split.folds,])$concordance
  # DR-NN-Cox
  gbsg.cox.m3.test.temp <- coxph(gbsg.surv_form.cox, data = gbsg.f.m3.temp$data_feature[split.folds,])
  gbsg.nfolds.cindex.nn.cox$DR[i] = concordance(gbsg.cox.m3.test.temp, newdata = gbsg.f.m3.temp$data_feature[-split.folds,])$concordance
  
  ## ----  NN-Cox(NP) ------
  # Apply time splitting
  gbsg.time.interval = 10
  gbsg.cox.var = c("duration", "event", paste0("feature.", c(1:gbsg.n_feature)))
  gbsg.f.m1.cox.temp = survSplit(Surv(duration, event)~., data = gbsg.f.m1.temp$data_feature[, gbsg.cox.var],
                                 cut = seq(0, max(gbsg.f.m1.temp$data_feature$duration), gbsg.time.interval)[-1], 
                                 episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  gbsg.f.m2.cox.temp = survSplit(Surv(duration, event)~., data = gbsg.f.m2.temp$data_feature[, gbsg.cox.var],
                                 cut = seq(0, max(gbsg.f.m1.temp$data_feature$duration), gbsg.time.interval)[-1], 
                                 episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  gbsg.f.m3.cox.temp = survSplit(Surv(duration, event)~., data = gbsg.f.m3.temp$data_feature[, gbsg.cox.var],
                                 cut = seq(0, max(gbsg.f.m1.temp$data_feature$duration), gbsg.time.interval)[-1], 
                                 episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  
  # Set up formula
  form = "Surv(tstart, tstop, event) ~ feature.1"
  for (f in 2: gbsg.n_feature){form = paste0(form, "+feature.", f)}
  set.seed(82); jj = sample(1:gbsg.n_feature, ceiling(gbsg.n_feature/2))

  for (j in jj){
    form = paste0(form, "+t.feature.",j)
    gbsg.f.m1.cox.temp[, paste0("t.feature.", j)] = gbsg.f.m1.cox.temp$tstart*gbsg.f.m1.cox.temp[,paste0("feature.", j)]
    gbsg.f.m2.cox.temp[, paste0("t.feature.", j)] = gbsg.f.m2.cox.temp$tstart*gbsg.f.m2.cox.temp[,paste0("feature.", j)]
    gbsg.f.m3.cox.temp[, paste0("t.feature.", j)] = gbsg.f.m3.cox.temp$tstart*gbsg.f.m3.cox.temp[,paste0("feature.", j)]
  }
  gbsg.surv_form = as.formula(form)
  
  # RW-NN-Cox(NP)
  gbsg.cox.m1.test.temp.np <- coxph(gbsg.surv_form, data = gbsg.f.m1.cox.temp[gbsg.f.m1.cox.temp$id %in% split.folds,])
  gbsg.nfolds.cindex.nn.cox.np$RW[i] = concordance(gbsg.cox.m1.test.temp.np, newdata = gbsg.f.m1.cox.temp[!gbsg.f.m1.cox.temp$id %in% split.folds,])$concordance
  # MI-NN-Cox(NP)
  gbsg.cox.m2.test.temp.np <- coxph(gbsg.surv_form, data = gbsg.f.m2.cox.temp[gbsg.f.m2.cox.temp$id %in% split.folds,])
  gbsg.nfolds.cindex.nn.cox.np$MI[i] = concordance(gbsg.cox.m2.test.temp.np, newdata = gbsg.f.m2.cox.temp[!gbsg.f.m2.cox.temp$id %in% split.folds,])$concordance
  # DR-NN-Cox(NP)  
  gbsg.cox.m3.test.temp.np <- coxph(gbsg.surv_form, data = gbsg.f.m3.cox.temp[gbsg.f.m3.cox.temp$id %in% split.folds,])
  gbsg.nfolds.cindex.nn.cox.np$DR[i] = concordance(gbsg.cox.m3.test.temp.np, newdata = gbsg.f.m3.cox.temp[!gbsg.f.m3.cox.temp$id %in% split.folds,])$concordance
  
  print(paste("Fold", i, "is done!"))
}

colMeans(gbsg.nfolds.cindex.nn.cox)
colMeans(gbsg.nfolds.cindex.nn.cox.np)


############################
##### ---- flchain --- #####
############################
##--- Import data
data("flchain")
flchain.data = na.omit(flchain[,!names(flchain) %in% c("chapter")])

## k-fold CV
nfolds = 5
set.seed(726)
flchain.folds = createFolds(1:nrow(flchain.data[flchain.data$futime!=0,]), k = nfolds, list = T, returnTrain = F)
flchain.nfolds.cindex.nn.cox = flchain.nfolds.cindex.nn.cox.np = data.frame(matrix(NA, ncol = 4, nrow = nfolds))
names(flchain.nfolds.cindex.nn.cox) = names(flchain.nfolds.cindex.nn.cox.np) = c("Classic_Cox", "RW", "MI", "DR")

for (i in 1:nfolds){
  # Categorical covariates
  flchain.data$sex = as.numeric(flchain.data$sex)-1
  flchain.data = flchain.data.classic = dummy_cols(flchain.data, select_columns = "flc.grp")
  flchain.data.classic = flchain.data.classic[flchain.data.classic$futime!=0,]
  
  # Remove subjects with time 0
  flchain.data.drop = flchain.data[flchain.data$futime!=0,]
  
  split.folds = c(1:nrow(flchain.data.drop))[-flchain.folds[[i]]]
  
  # Numerical covariates
  num_cov_flchain = c("age", "sample.yr", "kappa", "lambda", "creatinine")
  flchain.data[split.folds,num_cov_flchain] = apply(flchain.data[split.folds,num_cov_flchain], 2, standardize)
  flchain.data[-split.folds,num_cov_flchain] = apply(flchain.data[-split.folds,num_cov_flchain], 2, standardize)
  
  ## ----  Classic Cox ------
  flchain.cox.temp <- coxph(Surv(futime, death) ~ ., 
                            data = flchain.data.classic[split.folds,!names(flchain.data.classic) %in% c("flc.grp")])
  flchain.nfolds.cindex.nn.cox$Classic_Cox[i] = flchain.nfolds.cindex.nn.cox.np$Classic_Cox =
    concordance(flchain.cox.temp, newdata = flchain.data.classic[-split.folds,!names(flchain.data.classic) %in% c("flc.grp")])$concordance
  
  ##--- Transformation
  flchain.y.m1 = flchain.y.m2 = flchain.y.m3 = vector("logical", length = nrow(flchain.data.drop))
  # method 1: reweighing 
  flchain.data.m1 = reweighing(data = data.frame(time = flchain.data.drop[split.folds,]$futime, event = flchain.data.drop[split.folds,]$death))
  flchain.y.m1[split.folds] = flchain.data.m1$y.new.m1
  # method 2: mean imputation
  flchain.data.m2 = mean.imputation(data = data.frame(time = flchain.data.drop[split.folds,]$futime, event = flchain.data.drop[split.folds,]$death))
  flchain.y.m2[split.folds] = flchain.data.m2$y.new.m2
  # method 3: deviance residual
  flchain.data.m3 = deviance.residual(data = data.frame(time = flchain.data.drop[split.folds,]$futime, event = flchain.data.drop[split.folds,]$death))
  flchain.y.m3[split.folds] = flchain.data.m3$y.new.m3
  
  ##--- Feature extraction with NN
  flchain.x.var = names(flchain.data.drop)[1:8]
  
  flchain.n_feature = 10
  flchain.f.m1.temp = Surv_NN(data = cbind(flchain.data.drop, y.m1=flchain.y.m1), y.var = "y.m1", x.var = flchain.x.var, 
                              hidden.nodes = c(20, flchain.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear", 
                              split.folds = split.folds, batch.size = 64, epochs = 50)
  
  flchain.f.m2.temp = Surv_NN(data = cbind(flchain.data.drop, y.m2=flchain.y.m2), y.var= "y.m2", x.var = flchain.x.var,
                              hidden.nodes = c(20, flchain.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear",
                              split.folds = split.folds,batch.size = 64, epochs = 50)
  
  flchain.f.m3.temp = Surv_NN(data = cbind(flchain.data.drop, y.m3=flchain.y.m3), y.var= "y.m3", x.var = flchain.x.var, 
                              hidden.nodes = c(20, flchain.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear", 
                              split.folds = split.folds, batch.size = 64, epochs = 50) 
  
  ## ----  NN-Cox ------
  form.cox = "Surv(futime, death) ~ feature.1"
  for (j in 2: flchain.n_feature){form.cox = paste0(form.cox, "+feature.", j)}
  flchain.surv_form.cox = as.formula(form.cox)
  
  # RW-NN-Cox
  flchain.cox.m1.test.temp <- coxph(flchain.surv_form.cox, data = flchain.f.m1.temp$data_feature[split.folds,])
  flchain.nfolds.cindex.nn.cox$RW[i] = concordance(flchain.cox.m1.test.temp, newdata = flchain.f.m1.temp$data_feature[-split.folds,])$concordance
  # MI-NN-Cox
  flchain.cox.m2.test.temp <- coxph(flchain.surv_form.cox, data = flchain.f.m2.temp$data_feature[split.folds,])
  flchain.nfolds.cindex.nn.cox$MI[i] = concordance(flchain.cox.m2.test.temp, newdata = flchain.f.m2.temp$data_feature[-split.folds,])$concordance
  # DR-NN-Cox
  flchain.cox.m3.test.temp <- coxph(flchain.surv_form.cox, data = flchain.f.m3.temp$data_feature[split.folds,])
  flchain.nfolds.cindex.nn.cox$DR[i] = concordance(flchain.cox.m3.test.temp, newdata = flchain.f.m3.temp$data_feature[-split.folds,])$concordance
  
  ## ----  NN-Cox(NP) ------
  # Apply time splitting
  flchain.time.interval = 400
  flchain.cox.var = c("futime", "death", paste0("feature.", c(1:flchain.n_feature)))
  flchain.f.m1.cox.temp = survSplit(Surv(futime, death)~., data = flchain.f.m1.temp$data_feature[, flchain.cox.var],
                                    cut = seq(0, max(flchain.f.m1.temp$data_feature$futime), flchain.time.interval)[-1], 
                                    episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  flchain.f.m2.cox.temp = survSplit(Surv(futime, death)~., data = flchain.f.m2.temp$data_feature[, flchain.cox.var],
                                    cut = seq(0, max(flchain.f.m2.temp$data_feature$futime), flchain.time.interval)[-1], 
                                    episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  flchain.f.m3.cox.temp = survSplit(Surv(futime, death)~., data = flchain.f.m3.temp$data_feature[, flchain.cox.var],
                                    cut = seq(0, max(flchain.f.m3.temp$data_feature$futime), flchain.time.interval)[-1], 
                                    episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  
  # Set up formula
  form = "Surv(tstart, tstop, death) ~ feature.1"
  for (f in 2: flchain.n_feature){form = paste0(form, "+feature.", f)}
  set.seed(812); jj = sample(1:flchain.n_feature, ceiling(flchain.n_feature/2))
  
  for (j in jj){
    form = paste0(form, "+t.feature.",j)
    flchain.f.m1.cox.temp[, paste0("t.feature.", j)] = flchain.f.m1.cox.temp$tstart*flchain.f.m1.cox.temp[,paste0("feature.", j)]
    flchain.f.m2.cox.temp[, paste0("t.feature.", j)] = flchain.f.m2.cox.temp$tstart*flchain.f.m2.cox.temp[,paste0("feature.", j)]
    flchain.f.m3.cox.temp[, paste0("t.feature.", j)] = flchain.f.m3.cox.temp$tstart*flchain.f.m3.cox.temp[,paste0("feature.", j)]
  }
  flchain.surv_form = as.formula(form)
  
  # RW-NN-Cox(NP)
  flchain.cox.m1.test.temp.np <- coxph(flchain.surv_form, data = flchain.f.m1.cox.temp[flchain.f.m1.cox.temp$id %in% split.folds,])
  flchain.nfolds.cindex.nn.cox.np$RW[i] = concordance(flchain.cox.m1.test.temp.np, newdata = flchain.f.m1.cox.temp[!flchain.f.m1.cox.temp$id %in% split.folds,])$concordance
  # MI-NN-Cox(NP)
  flchain.cox.m2.test.temp.np <- coxph(flchain.surv_form, data = flchain.f.m2.cox.temp[flchain.f.m2.cox.temp$id %in% split.folds,])
  flchain.nfolds.cindex.nn.cox.np$MI[i] = concordance(flchain.cox.m2.test.temp.np, newdata = flchain.f.m2.cox.temp[!flchain.f.m2.cox.temp$id %in% split.folds,])$concordance
  # DR-NN-Cox(NP)
  flchain.cox.m3.test.temp.np <- coxph(flchain.surv_form, data = flchain.f.m3.cox.temp[flchain.f.m3.cox.temp$id %in% split.folds,])
  flchain.nfolds.cindex.nn.cox.np$DR[i] = concordance(flchain.cox.m3.test.temp.np, newdata = flchain.f.m3.cox.temp[!flchain.f.m3.cox.temp$id %in% split.folds,])$concordance
  
  print(paste("Fold", i, "is done!"))
}

colMeans(flchain.nfolds.cindex.nn.cox)
colMeans(flchain.nfolds.cindex.nn.cox.np)
