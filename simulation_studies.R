library(survival)
library(fastDummies)
library(tensorflow)
library(keras)
library(caret)
library(pbapply)
library(tidyverse)
setwd("~")
source("functions.R")

# Import simulated data
N = 500 # available options: 500, 5000
type = "prop" # available options: "prop", "nonlinear", "nonprop"
censor.rate = 0.3 # available options: 0.3, 0.6
load(paste0("datasets/sim_datasets/sim_", type,"_N",N, "_censor", censor.rate, ".RData"))

simdata = sim$data[,-1]
simdata[simdata$time==0,]
names(simdata)
x = sim$x
beta = sim$beta
Tmax = max(sim$data$time)
cov.no = dim(sim$x)[2]

## 100 replicates
nrep = 100
sim.reps = list()
set.seed(77)
for (i in 1:nrep){sim.reps[[i]] = sample(1:N, ceiling(N/2), replace = F)}
sim.nrep.cindex.nn.cox = sim.nrep.cindex.nn.cox.np = data.frame(matrix(NA, ncol = 4, nrow = nrep))
names(sim.nrep.cindex.nn.cox) = names(sim.nrep.cindex.nn.cox.np) = c("Classic_Cox", "RW", "MI", "DR")

for (i in 1:nrep){
  split.train = c(1:N)[sim.reps[[i]]]
  
  # Categorical covariates
  # num_cov_sim = c("x1")
  # simdata = dummy_cols(simdata, select_columns = num_cov_sim)
  
  # Numerical covariates
  num_cov_sim = colnames(x)
  simdata[split.train,num_cov_sim] = apply(simdata[split.train,num_cov_sim], 2, standardize)
  simdata[-split.train,num_cov_sim] = apply(simdata[-split.train,num_cov_sim], 2, standardize)
  
  ## ----  Classic Cox ------
  sim.cox.temp <- coxph(Surv(time, event) ~ ., data = simdata[split.train,])
  sim.nrep.cindex.nn.cox$Classic_Cox[i] = sim.nrep.cindex.nn.cox.np$Classic_Cox[i] = 
    concordance(sim.cox.temp, newdata = simdata[-split.train,])$concordance
  
  # Remove subjects with time 0
  sim.data.drop = simdata[simdata$time!=0,]
  
  ##--- Transformation
  sim.y.m1 = sim.y.m2 = sim.y.m3 = vector("logical", length = nrow(sim.data.drop))
  # method 1: reweighing 
  sim.data.m1 = reweighing(data = data.frame(time = sim.data.drop[split.train,]$time, event = sim.data.drop[split.train,]$event))
  sim.y.m1[split.train] = sim.data.m1$y.new.m1
  # method 2: mean imputation
  sim.data.m2 = mean.imputation(data = data.frame(time = sim.data.drop[split.train,]$time, event = sim.data.drop[split.train,]$event))
  sim.y.m2[split.train] = sim.data.m2$y.new.m2
  # method 3: deviance residual 
  sim.data.m3 = deviance.residual(data = data.frame(time = sim.data.drop[split.train,]$time, event = sim.data.drop[split.train,]$event))
  sim.y.m3[split.train] = sim.data.m3$y.new.m3
  
  ##--- Feature extraction with NN
  sim.x.var = colnames(x)
  
  sim.n_feature = 10
  sim.f.m1.temp = Surv_NN(data = cbind(sim.data.drop, y.m1=sim.y.m1), y.var = "y.m1", x.var = sim.x.var, 
                          hidden.nodes = c(20, sim.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear", 
                          split.folds = split.train, batch.size = 48, epochs = 100)
  
  sim.f.m2.temp = Surv_NN(data = cbind(sim.data.drop, y.m2=sim.y.m2), y.var= "y.m2", x.var = sim.x.var,
                          hidden.nodes = c(20, sim.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear",
                          split.folds = split.train,batch.size = 48, epochs = 100)

  sim.f.m3.temp = Surv_NN(data = cbind(sim.data.drop, y.m3=sim.y.m3), y.var= "y.m3", x.var = sim.x.var, 
                          hidden.nodes = c(20, sim.n_feature), activations = c('sigmoid', 'sigmoid'), output.activation = "linear", 
                          split.folds = split.train, batch.size = 48, epochs = 100) 
  
  ## ----  NN-Cox ------
  form.cox = "Surv(time, event) ~ feature.1"
  for (j in 2: sim.n_feature){form.cox = paste0(form.cox, "+feature.", j)}
  sim.surv_form.cox = as.formula(form.cox)
  
  # RW-NN-Cox
  sim.cox.m1.test.temp <- coxph(sim.surv_form.cox, data = sim.f.m1.temp$data_feature[split.train,])
  sim.nrep.cindex.nn.cox$RW[i] = concordance(sim.cox.m1.test.temp, newdata = sim.f.m1.temp$data_feature[-split.train,])$concordance
  # MI-NN-Cox
  sim.cox.m2.test.temp <- coxph(sim.surv_form.cox, data = sim.f.m2.temp$data_feature[split.train,])
  sim.nrep.cindex.nn.cox$MI[i] = concordance(sim.cox.m2.test.temp, newdata = sim.f.m2.temp$data_feature[-split.train,])$concordance
  # DR-NN-Cox
  sim.cox.m3.test.temp <- coxph(sim.surv_form.cox, data = sim.f.m3.temp$data_feature[split.train,])
  sim.nrep.cindex.nn.cox$DR[i] = concordance(sim.cox.m3.test.temp, newdata = sim.f.m3.temp$data_feature[-split.train,])$concordance
  
  
  ## ----  NN-Cox(NP) ------
  # Apply time splitting
  sim.time.interval = 2
  sim.cox.var = c("time", "event", paste0("feature.", c(1:sim.n_feature)))
  sim.f.m1.cox.temp = survSplit(Surv(time, event)~., data = sim.f.m1.temp$data_feature[, sim.cox.var],
                                cut = seq(0, max(sim.f.m1.temp$data_feature$time), sim.time.interval)[-1], 
                                episode = "tgroup", start = "tstart", end = "tstop", id = "id")

  sim.f.m2.cox.temp = survSplit(Surv(time, event)~., data = sim.f.m2.temp$data_feature[, sim.cox.var],
                                cut = seq(0, max(sim.f.m2.temp$data_feature$time), sim.time.interval)[-1], 
                                episode = "tgroup", start = "tstart", end = "tstop", id = "id")

  sim.f.m3.cox.temp = survSplit(Surv(time, event)~., data = sim.f.m3.temp$data_feature[, sim.cox.var],
                                cut = seq(0, max(sim.f.m3.temp$data_feature$time), sim.time.interval)[-1], 
                                episode = "tgroup", start = "tstart", end = "tstop", id = "id")
  
  # Set up formula
  form = "Surv(tstart, tstop, event) ~ feature.1"
  for (f in 2: sim.n_feature){form = paste0(form, "+feature.", f)}
  set.seed(238); jj = sample(1:sim.n_feature, ceiling(sim.n_feature/3))
  for (j in jj){
    form = paste0(form, "+t.feature.",j)
    sim.f.m1.cox.temp[, paste0("t.feature.", j)] = sim.f.m1.cox.temp$tstart*sim.f.m1.cox.temp[,paste0("feature.", j)]
    sim.f.m2.cox.temp[, paste0("t.feature.", j)] = sim.f.m2.cox.temp$tstart*sim.f.m2.cox.temp[,paste0("feature.", j)]
    sim.f.m3.cox.temp[, paste0("t.feature.", j)] = sim.f.m3.cox.temp$tstart*sim.f.m3.cox.temp[,paste0("feature.", j)]
  }
  sim.surv_form = as.formula(form)
  
  # RW-NN-Cox(NP)
  sim.cox.m1.test.temp.np <- coxph(sim.surv_form, data = sim.f.m1.cox.temp[sim.f.m1.cox.temp$id %in% split.train,])
  sim.nrep.cindex.nn.cox.np$RW[i] = concordance(sim.cox.m1.test.temp.np, newdata = sim.f.m1.cox.temp[!sim.f.m1.cox.temp$id %in% split.train,])$concordance
  # MI-NN-Cox(NP)
  sim.cox.m2.test.temp.np <- coxph(sim.surv_form, data = sim.f.m2.cox.temp[sim.f.m2.cox.temp$id %in% split.train,])
  sim.nrep.cindex.nn.cox.np$MI[i] = concordance(sim.cox.m2.test.temp.np, newdata = sim.f.m2.cox.temp[!sim.f.m2.cox.temp$id %in% split.train,])$concordance
  # DR-NN-Cox(NP)
  sim.cox.m3.test.temp.np <- coxph(sim.surv_form, data = sim.f.m3.cox.temp[sim.f.m3.cox.temp$id %in% split.train,])
  sim.nrep.cindex.nn.cox.np$DR[i] = concordance(sim.cox.m3.test.temp.np, newdata = sim.f.m3.cox.temp[!sim.f.m3.cox.temp$id %in% split.train,])$concordance
  
  print(paste("Replicate", i, "is done!"))
}

colMeans(sim.nrep.cindex.nn.cox)
sapply(sim.nrep.cindex.nn.cox, sd)

colMeans(sim.nrep.cindex.nn.cox.np)
sapply(sim.nrep.cindex.nn.cox.np, sd)
