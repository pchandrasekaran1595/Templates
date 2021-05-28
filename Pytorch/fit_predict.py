import torch
from torch import nn

import os
import numpy as np
from sklearn.metrics import __________

from time import time

def breaker():
    print("\n" + 50*"-" + "\n")
    
 
###############################################################################################
"""
    General fit function used to train a single model.
"""

def fit_(model=None, optimizer=None, scheduler=None, epochs=None, early_stopping_patience=None, 
         trainloader=None, validloader=None, trainlabels=None, validlabels=None, save_to_file=False,
         criterion=None, device=None, checkpoint_freq=1, path=None, verbose=True):
    
    breaker()
    print("Training ...")
    breaker()
    
    if trainlabels and validlabels:
        LBS = {"train" : trainlabels, "valid" : validlabels}

    if save_to_file:
        file = open(os.path.join(path, "Metrics.txt"), "w")
    
    model.to(device)
    
    DLS = {"train" : trainloader, "valid" : validloader}
    bestLoss = {"train" : np.inf, "valid" : np.inf}
    # bestMetric = {"train" : 0.0, "valid" : 0.0}
    
    Losses = []
    # Metrics = []
    
    start_time = time()
    for e in range(epochs):
        e_st = time()
        
        epochLoss = {"train" : 0.0, "valid" : 0.0}
        # epochMetric = {"train" : 0.0, "valid" : 0.0}
        
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
                
            lossPerPass = []
            # metricPerPass = []
            
            for X, y in DLS[phase]:
                X = X.to(device)
                if y.dtype == torch.int64:
                    y = y.to(device).view(-1)
                else:
                    y = y.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output, _, _ = model(X)
                    loss = criterion(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item())
                # metricPerPass
            epochLoss[phase] = np.mean(np.array(lossPerPass))
            # epochMetric[phase] = np.mean(np.array(metricPerPass))
        Losses.append(epochLoss)
        # Metrics.append(epochMetric)

        # # Use if it is needed to assuredly save state every epoch
        # torch.save({"model_state_dict" : model.state_dict(),
        #             "optim_state_dict" : optimizer.state_dict()},
        #             os.path.join(path, "Epoch_{}.pt".format(e+1)))
        
        if (e+1) % checkpoint_freq == 0:
            torch.save({"model_state_dict" : model.state_dict(),
                        "optim_state_dict" : optimizer.state_dict()},
                        os.path.join(path, "Epoch_{}.pt".format(e+1)))
        
        if early_stopping_patience:
            if epochLoss["valid"] < bestLoss["valid"]: # or epochMetric["valid"] __ bestMetric["valid"]:
                bestLoss = epochLoss
                bestLossEpoch = e+1
                torch.save({"model_state_dict" : model.state_dict(),
                            "optim_state_dict" : optimizer.state_dict()},
                            os.path.join(path, "Epoch_{}.pt".format(e+1)))
                early_stopping_step = 0
            else:
                early_stopping_step += 1
                if early_stopping_step > early_stopping_patience:
                    print("Early Stopping at Epoch {}".format(e+1))
                    break
                    
        if scheduler:
            scheduler.step()
            # scheduler.step(epochLoss["valid"])
         
        if epochLoss["valid"] < bestLoss["valid"]:
            bestLoss = epochLoss
            bestLossEpoch = e+1
            torch.save({"model_state_dict" : model.state_dict(),
                        "optim_state_dict" : optimizer.state_dict()},
                        os.path.join(path, "Epoch_{}.pt".format(e+1)))
            
        if verbose:
            print("Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} | Time: {:.2f} seconds".format(e+1, 
                                                                                                      epochLoss["train"], epochLoss["valid"], 
                                                                                                      time()-e_st))
            # Add metrics if used
            
        if save_to_file:
            text = "Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} | Time: {:.2f} seconds\n".format(e+1, 
                                                                                                         epochLoss["train"], epochLoss["valid"],
                                                                                                         time() - e_st)
            file.write(text)
            file.close()
            # Add metrics if used
            
    # Include Best Validation Metric Epoch if Used
    breaker()
    print("-----> Best Validation Loss at Epoch {}".format(bestLossEpoch))
    breaker()
    print("Time Taken [{} Epochs] : {:.2f} minutes".format(epochs, (time()-start_time)/60))
    breaker()
    print("Training Complete")
    breaker()

    if save_to_file:
        text_1 = "\n-----> Best Validation Loss at Epoch {}\n".format(bestLossEpoch)
        text_2 = "Time Taken [{} Epochs] : {:.2f} minutes\n".format(len(Losses), (time() - start_time) / 60)

        file.write(text_1)
        file.write(text_2)
        file.close()

    # Return Appropriate Variables
    return Losses, bestLossEpoch
            
###############################################################################################

"""
1. Used for Binary/MultiLabel Classification and Regression Problems.
2. Use torch.sigmoid(model(X)) if nn.BCEWithLogitsLoss() is used.
"""

def predict_1(model=None, dataloader=None, device=None, NUM_LABELS=None, mode="valid", path=None):
    if path:
        model.load_state_dict(torch.load(path, mao_selection=device)["model_state_dict"])

    model.eval()
    model.to(device)

    if not NUM_LABELS:
        y_pred = torch.zeros(1, 1).to(device)
    else:
        y_pred = torch.zeros(1, NUM_LABELS).to(device)

    if mode == "valid":
        for X, y in dataloader: # for X, y in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = model(X) # torch.sigmoid(model(X))
            if not NUM_LABELS:
                y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
            else:
                y_pred = torch.cat((y_pred, output), dim=0)
    elif mode == "test":
        for X in dataloader: # for X, y in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = model(X) # torch.sigmoid(model(X))
            if not NUM_LABELS:
                y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
            else:
                y_pred = torch.cat((y_pred, output), dim=0)
    else:
        breaker()
        print("Invalid Mode")
        breaker()
                   
    return y_pred[1:].detach().cpu().numpy()

# ********************************************************************************************* #

"""
1. Used for MultiClass Classification Problems.
2. Use torch.exp(model(X)) if nn.NLLLoss() is used.
"""

def predict_2(model=None, dataloader=None, device=None, mode="valid", path=None):
    if path:
        model.load_state_dict(torch.load(path, map_selection=device)["model_state_dict"])

    model.eval()
    model.to(device)

    y_pred = torch.zeros(1, 1).to(device)
    
    if mode == "valid":
        for X, y in dataloader:  # for X, y in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = torch.argmax(model(X), dim=1) # output = torch.argmax(torch.exp(model(X)), dim=1)
            y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
    elif mode == "test":
        for X in dataloader:  # for X, y in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = torch.argmax(model(X), dim=1) # output = torch.argmax(torch.exp(model(X)), dim=1)
            y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
    else:
        breaker()
        print("Invalid Mode")
        breaker()

    return y_pred[1:].detach().cpu().numpy().astype("int")

# ********************************************************************************************* #
