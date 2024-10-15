
import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from sklearn.model_selection import KFold, train_test_split
import seaborn as sns
#from sklearn.inspection import permutation_importance

os.chdir('/Users/hiro/DS-ML')
from DS_ML_Data import clean
from DS_ML_Data_2 import clean2

import pickle

#dd = clean2()


random.seed(42)
d, train_df, test_df = clean()
x, y, p, ce, ide = [torch.tensor(train_df['x'].values.astype(np.float32)).unsqueeze(1), torch.tensor(train_df['y'].values.astype(np.float32)).unsqueeze(1), torch.tensor(train_df['p'].values.astype(np.float32)).unsqueeze(1), torch.tensor(train_df['ce'].values.astype(np.float32)).unsqueeze(1), torch.tensor(train_df['subject_global'].values.astype(np.int64)).unsqueeze(1)]
x2, y2, p2, ce2, ide2 = [torch.tensor(test_df['x'].values.astype(np.float32)).unsqueeze(1), torch.tensor(test_df['y'].values.astype(np.float32)).unsqueeze(1), torch.tensor(test_df['p'].values.astype(np.float32)).unsqueeze(1), torch.tensor(test_df['ce'].values.astype(np.float32)).unsqueeze(1), torch.tensor(test_df['subject_global'].values.astype(np.int64)).unsqueeze(1)]

criterion = nn.MSELoss()
batch_size = 8


def graph_prob(model, function, name, name2):
    
    p_values = torch.linspace(0, 1, 1000).unsqueeze(1)
    model.eval()
    with torch.no_grad():  # We do not need to track gradients here
        pi_p = function(p_values)
        
    p_values = p_values.numpy().flatten()
    pi_p = pi_p.numpy().flatten()
    
    data = pd.DataFrame({'p (Probability)': p_values, name: pi_p})
    
    plt.figure(figsize=(8, 6))
    
    # Adding the 45-degree line with a subtle dashed linestyle
    plt.plot(p_values, p_values, linestyle='--', color='black', label='45° Line (π(p) = p)', linewidth=1)
    
    # Adding vertical lines at x = 1/8 and x = 7/8 with subtle dotted linestyles
    #plt.axvline(x=1/8, linestyle=':', color='black', label='p = 1/8', linewidth=1)
    #plt.axvline(x=7/8, linestyle='-.', color='black', label='p = 7/8', linewidth=1)
    
    tick_values = [i/8 for i in range(9)]  # 0, 1/8, 2/8, ..., 1
    tick_labels = [f'{i/8}' for i in range(9)]  # '0/8', '1/8', ..., '8/8'
    plt.xticks(ticks=tick_values, labels=tick_labels)
    
    plt.yticks(ticks=tick_values, labels=tick_labels)
    
    # Plotting the learned probability weighting function with a distinct dark green color
    sns.lineplot(x='p (Probability)', y=name, data=data, linewidth=2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Set y-limit slightly above the max value for visibility


    # Setting labels and title
    plt.title(name, fontsize=14)
    plt.xlabel('Probability p', fontsize=12)
    plt.ylabel(name, fontsize=12)

    # Removing the grid for a cleaner look
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    
    filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
    plt.savefig(filename, format='pdf')
    
    plt.show()

#%%
# Model 1: Prospect Theory with just linear utility

###




class ProspectTheory(nn.Module):
    def __init__(self):
        super(ProspectTheory, self).__init__()    
        self.probability_weighting = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
    def forward(self, x, y, p):
        pi_p = self.probability_weighting(p)
        u_ce = pi_p *x + (1 - pi_p) * y
        return u_ce

batch_size = 8
start_time_1 = time.time()       
model_1 = ProspectTheory() 

patience = 10
early_stop = False


model_1 = ProspectTheory()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)




    




def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0
        for batch in train_loader:
            x, y, p, ce = batch 
            
            optimizer.zero_grad()
            predicted_ce = model(x, y, p)
            
            loss = criterion(predicted_ce, ce)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y, p, ce = batch
                predicted_ce = model(x, y, p)
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break
    return best_model_state, best_val_loss
        
        
    




dataset = TensorDataset(x,y,p,ce)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_1 = []

best_model_state_1 = None
best_fold_val_loss = float('inf')


#"""
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_1 = ProspectTheory()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_1, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_1.append(fold_val_loss) 

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_1 = fold_best_model_state


# Load the best model for evaluation
best_model_1 = ProspectTheory()
best_model_1.load_state_dict(best_model_state_1)
best_model_1.eval()

torch.save(best_model_state_1, 'ProspectTheory.pth')

with open('losses_per_fold_1.pkl', 'wb') as f:
    pickle.dump(losses_per_fold_1, f)

#"""
best_model_1 = ProspectTheory()
best_model_1.load_state_dict(torch.load('ProspectTheory.pth'))
best_model_1.eval()

with open('losses_per_fold_1.pkl', 'rb') as f:
    losses_per_fold_1 = pickle.load(f)

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    
    with torch.no_grad():  # No gradient updates needed for evaluation
        for batch in test_loader:
            x, y, p, ce = batch
            predicted_ce = model(x, y, p)
            predictions.append(predicted_ce)
            actuals.append(ce)
    
    # Convert lists to tensors
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
    
    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Absolute Error on Test Set: {mae.item():.4f}')
    
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)
    
    return mse, mae, comparison

test_dataset = TensorDataset(x2,y2,p2,ce2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('Evaluating model on Test Dataset')
mse_1, mae_1, comparison_1 = evaluate_model(best_model_1, test_loader)


end_time_1 = time.time()
time_diff_1 = abs(start_time_1 - end_time_1)/60

print("PT")
print(f"Time taken (minutes): {time_diff_1:.2f}") #7.83 minutes
print(f"Losses per fold: {losses_per_fold_1}") #[16.06855511242461, 15.251242824176765, 15.542200811193624, 15.84541655906479, 15.695355646398818]
print(f"MSE {mse_1}")   #15.750




name2 = 'Model 1' 
#### Analysis
graph_prob(model_1, model_1.probability_weighting,'Probability Weighting π(p)', name2 = 'Model 1')


predicted_ce = comparison_1[:, 0]
actual_ce = comparison_1[:, 1]

plt.scatter(actual_ce, predicted_ce, alpha=0.5)
plt.xlabel('Actual CE')
plt.ylabel('Predicted CE')
plt.title('Predicted vs Actual Certainty Equivalents')
plt.plot([min(actual_ce), max(actual_ce)], [min(actual_ce), max(actual_ce)], color='red', linestyle='--')  # y=x line
plt.grid(True)
plt.show()

epochs = range(1, len(losses_per_fold_1) * 10, 10)
plt.plot(epochs, losses_per_fold_1, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


name = 'Distribution of Residuals' 
residuals = predicted_ce - actual_ce
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residual (Predicted CE - Actual CE)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()


#####



#%%
# Model 2: Cumulative Prospect Theory




class CumulativeProspectTheory(nn.Module):
    def __init__(self):
        super(CumulativeProspectTheory, self).__init__()
    
        self.probability_weighting_gains = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.probability_weighting_losses = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
        
    def forward(self, x, y, p):
        pi_p_x = torch.where(
            (x > 0) & (y >= 0),  # Gains for both outcomes
            self.probability_weighting_gains(p), 
            torch.where(
                (x < 0) & (y <= 0),  # Losses for both outcomes
                self.probability_weighting_losses(p), 
                self.probability_weighting_gains(p)  # Mixed outcomes
            )
        )

        pi_p_y = torch.where(
            (x > 0) & (y >= 0),  # Gains for both
            1 - pi_p_x,
            torch.where(
                (x < 0) & (y <= 0),  # Losses for both
                1 - pi_p_x,
                self.probability_weighting_losses(1 - p)  # Mixed outcomes
            )
        )

            

        
        u_ce = pi_p_x * x + pi_p_y * y
        
        return u_ce


start_time_2 = time.time()        
model_2 = CumulativeProspectTheory()
batch_size = 8

patience = 10
early_stop = False

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model_2.parameters(), lr=0.01)







def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0
        for batch in train_loader:
            x, y, p, ce = batch 
            
            
            optimizer.zero_grad()
            predicted_ce = model(x, y, p)
            
            loss = criterion(predicted_ce, ce)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y, p, ce = batch
                predicted_ce = model(x, y, p)
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break
    return best_model_state, best_val_loss
        
        


# Create a TensorDataset
dataset = TensorDataset(x,y,p,ce)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_2 = []

best_model_state_2 = None
best_fold_val_loss = float('inf')

#train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


#"""
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_2 = CumulativeProspectTheory()
    optimizer = torch.optim.Adam(model_2.parameters(), lr=0.01)
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_2, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_2.append(fold_val_loss) 

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_2 = fold_best_model_state


# Load the best model for evaluation
best_model_2 = CumulativeProspectTheory()
best_model_2.load_state_dict(best_model_state_2)



torch.save(best_model_state_2, 'cumulative_prospect_theory_model.pth')

with open('losses_per_fold_2.pkl', 'wb') as f:
    pickle.dump(losses_per_fold_2, f)
#"""

best_model_2 = CumulativeProspectTheory()
best_model_2.load_state_dict(torch.load('cumulative_prospect_theory_model.pth'))
best_model_2.eval()

with open('losses_per_fold_2.pkl', 'rb') as f:
    losses_per_fold_2 = pickle.load(f)


test_dataset = TensorDataset(x2,y2,p2,ce2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    
    with torch.no_grad():  # No gradient updates needed for evaluation
        for batch in test_loader:
            x, y, p, ce = batch
            predicted_ce = model(x, y, p)
            predictions.append(predicted_ce)
            actuals.append(ce)
    
    # Convert lists to tensors
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    # Calculate MSE between predicted and actual CE values
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
    
    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Average Error on Test Set: {mae.item():.4f}')
    # Print the comparison
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)
    
    return mse, mae, comparison


print('\nEvaluating model on Test Dataset')
mse_2, mae_2, comparison_2 = evaluate_model(best_model_2, test_loader)


end_time_2 = time.time()
time_diff_2 = abs(start_time_2 - end_time_2)/60


print("CPT")
print(f"Time taken (minutes): {time_diff_2:.2f}") 
print(f"Losses per fold: {losses_per_fold_2}") 
print(f"MSE {mse_2}")  #17.21717



def graph_prob2(model, function_gains, function_losses, name_gains, name_losses, name2):
    p_values = torch.linspace(0, 1, 1000).unsqueeze(1)
    model.eval()
    with torch.no_grad():  # We do not need to track gradients here
        pi_p_gains = function_gains(p_values)
        pi_p_losses = function_losses(p_values)
        
    p_values = p_values.numpy().flatten()
    pi_p_gains = pi_p_gains.numpy().flatten()
    pi_p_losses = pi_p_losses.numpy().flatten()
    
    data_gains = pd.DataFrame({'p (Probability)': p_values, name_gains: pi_p_gains})
    data_losses = pd.DataFrame({'p (Probability)': p_values, name_losses: pi_p_losses})
    
    
    plt.figure(figsize=(8, 6))
    
    # Adding the 45-degree line with a subtle dashed linestyle
    plt.plot(p_values, p_values, linestyle='--', color='black', label='45° Line (π(p) = p)', linewidth=1)
    

    
    tick_values = [i/8 for i in range(9)]  # 0, 1/8, 2/8, ..., 1
    tick_labels = [f'{i/8}' for i in range(9)]  # '0/8', '1/8', ..., '8/8'
    plt.xticks(ticks=tick_values, labels=tick_labels)
    
    plt.yticks(ticks=tick_values, labels=tick_labels)
    
    # Plotting the learned probability weighting function with a distinct dark green color
    sns.lineplot(x='p (Probability)', y=name_gains, data=data_gains, color='darkgreen', linewidth=2, label=name_gains)
    sns.lineplot(x='p (Probability)', y=name_losses, data=data_losses, color='darkred', linewidth=2, label=name_losses)

    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Set y-limit slightly above the max value for visibility


    # Setting labels and title
    plt.title('Probability Weighting Functions for Gains and Losses', fontsize=14)
    plt.xlabel('Probability p', fontsize=12)
    plt.ylabel('Weighting Function Value', fontsize=12)

    
    # Removing the grid for a cleaner look
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    
    name = 'Probability Weighting Positive and Negative'
    filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
    plt.savefig(filename, format='pdf')
    
    plt.show()


graph_prob2(best_model_2, best_model_2.probability_weighting_gains, best_model_2.probability_weighting_losses, 'Probability Weighting Gains π(p)', 'Probability Weighting Losses π(p)', name2 = 'Model 2')




predicted_ce_2 = comparison_2[:, 0]
actual_ce_2 = comparison_2[:, 1]

plt.scatter(actual_ce_2, predicted_ce_2, alpha=0.5, edgecolor='white')
plt.xlabel('Actual CE')
plt.ylabel('Predicted CE')
plt.title('Predicted vs Actual Certainty Equivalents')
plt.plot([min(actual_ce_2), max(actual_ce_2)], [min(actual_ce_2), max(actual_ce_2)], color='red', linestyle='--')  # y=x line
plt.grid(True)
plt.show()

epochs = range(1, len(losses_per_fold_2) * 10, 10)
plt.plot(epochs, losses_per_fold_2, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

name = 'Distribution of Residuals' 
name2 = 'Model 2'
residuals = predicted_ce_2 - actual_ce_2
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residual (Predicted CE - Actual CE)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()








#####


#%%
# Model 3: Neural Network with no structure




start_time_3 = time.time()






class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 1)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

patience = 10
early_stop = False
criterion = nn.MSELoss()
model_3 = NeuralNetwork()
optimizer = torch.optim.Adam(model_3.parameters(), lr=0.01)
batch_size = 8



def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0
        for batch in train_loader:
            x, y, p, ce = batch 
            
            optimizer.zero_grad()
            var = torch.cat((x, y, p), dim=1)
            predicted_ce = model(var)
            
            loss = criterion(predicted_ce, ce)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y, p, ce = batch
                var = torch.cat((x, y, p), dim=1)
                predicted_ce = model(var)
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break
    return best_model_state, best_val_loss




#"""
# Create a TensorDataset
dataset = TensorDataset(x,y,p,ce)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_3 = []

best_model_state_3 = None
best_fold_val_loss = float('inf')

#train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_3 = NeuralNetwork()
    optimizer = torch.optim.Adam(model_3.parameters(), lr=0.01)
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_3, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_3.append(fold_val_loss) 

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_3 = fold_best_model_state


# Load the best model for evaluation
best_model_3 = NeuralNetwork()
best_model_3.load_state_dict(best_model_state_3)




torch.save(best_model_state_3, 'NeuralNetwork.pth')

with open('losses_per_fold_3.pkl', 'wb') as f:
    pickle.dump(losses_per_fold_3, f)
#"""

best_model_3 = NeuralNetwork()
best_model_3.load_state_dict(torch.load('NeuralNetwork.pth'))
best_model_3.eval()

with open('losses_per_fold_3.pkl', 'rb') as f:
    losses_per_fold_3 = pickle.load(f)


# Concatenate x2, y2, and p2 into a single tensor along the second dimension (i.e., column-wise)
features = torch.cat((x2, y2, p2), dim=1)

# Create the dataset with the concatenated features and the ce (certainty equivalent) as the target
test_dataset = TensorDataset(features, ce2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    
    with torch.no_grad():  # No gradient updates needed for evaluation
        for batch in test_loader:
            features, ce = batch
            predicted_ce = model(features)
            predictions.append(predicted_ce)
            actuals.append(ce)
    
    # Convert lists to tensors
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    # Calculate MSE between predicted and actual CE values
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
    
    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mae.item():.4f}')
    
    # Print the comparison
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)
    
    return mse, mae, comparison

print('\nEvaluating model on Test Dataset')
mse_3, mae_3, comparison_3 = evaluate_model(best_model_3, test_loader)


end_time_3 = time.time()
time_diff_3 = abs(start_time_3 - end_time_3)/60

print("NN")
print(f"Time taken (minutes): {time_diff_3:.2f}") #7.86
print(f"Losses per fold: {losses_per_fold_3}") #[16.06855511242461, 15.251242824176765, 15.542200811193624, 15.84541655906479, 15.695355646398818]
print(f"MSE {mse_3}")   #16.035018920898438


epochs = range(1, len(losses_per_fold_3) * 10, 10)
plt.plot(epochs, losses_per_fold_3, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()




predicted_ce_3 = comparison_3[:, 0]
actual_ce_3 = comparison_3[:, 1]

plt.scatter(actual_ce, predicted_ce_3, alpha=0.5, edgecolor='white')
plt.xlabel('Actual CE')
plt.ylabel('Predicted CE')
plt.title('Predicted vs Actual Certainty Equivalents')
plt.plot([min(actual_ce), max(actual_ce)], [min(actual_ce), max(actual_ce)], color='red', linestyle='--')  # y=x line
plt.grid(True)
plt.show()

residuals = predicted_ce_3 - actual_ce
plt.hist(residuals, bins=30, alpha=0.7, color='skyblue')
plt.xlabel('Residual (Predicted CE - Actual CE)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)
plt.show()

epochs = range(1, len(losses_per_fold_3) * 10, 10)
plt.plot(epochs, losses_per_fold_3, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()




#%%
# Model 4: Neural Network with individual identification





start_time_4 = time.time()




class NeuralNetwork_2(nn.Module):
    def __init__(self):
        super(NeuralNetwork_2, self).__init__()
        # Embedding layer for the ID data
        self.embedding = nn.Embedding(num_embeddings=2940, embedding_dim=32)  # 2939 unique IDs, embedding size of 32

        # Linear layers: adjust input size to include x, y, p + embedding dimension
        self.fc1 = nn.Linear(32 + 3, 128)  # 32 for embedding, 3 for x, y, p
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, ids, x, y, p):
        # Get embedding for IDs
        embedded_ids = self.embedding(ids)  # Shape: (batch_size, embedding_dim)

        # Concatenate embedding with numerical features (x, y, p)
        other_features = torch.cat((x, y, p), dim=1)  # Shape: (batch_size, 3)
        combined_input = torch.cat((embedded_ids, other_features), dim=1)  # Shape: (batch_size, 32 + 3)

        # Forward pass through fully connected layers
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, df):
        self.ids = torch.tensor(df['subject_global'].values, dtype=torch.long)  # Convert IDs to integers
        self.x = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)
        self.p = torch.tensor(df['p'].values, dtype=torch.float32).unsqueeze(1)
        self.ce = torch.tensor(df['ce'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.x[idx], self.y[idx], self.p[idx], self.ce[idx]


# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids, x, y, p, ce = batch
            
            optimizer.zero_grad()
            predicted_ce = model(ids, x, y, p)
            loss = criterion(predicted_ce, ce)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                ids, x, y, p, ce = batch
                predicted_ce = model(ids, x, y, p)
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            #print(f"Getting better! Epoch {epoch}")
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break
        print(epoch)
    return best_model_state, best_val_loss




patience = 10
early_stop = False
criterion = nn.MSELoss()
model_4 = NeuralNetwork_2()
optimizer = torch.optim.Adam(model_4.parameters(), lr=0.01)
batch_size = 8


train_dataset = CustomDataset(train_df)
dataset = train_dataset

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_4 = []

best_model_state_4 = None
best_fold_val_loss = float('inf')

    #"""
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_4 = NeuralNetwork_2()
    optimizer = torch.optim.Adam(model_4.parameters(), lr=0.01)
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_4, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_4.append(fold_val_loss) 

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_4 = fold_best_model_state


# Load the best model for evaluation
best_model_4 = NeuralNetwork_2()
best_model_4.load_state_dict(best_model_state_4)

torch.save(best_model_state_4, 'NeuralNetwork_2.pth')

with open('losses_per_fold_4.pkl', 'wb') as f:
    pickle.dump(losses_per_fold_4, f)

#"""

best_model_4 = NeuralNetwork_2()
best_model_4.load_state_dict(torch.load('NeuralNetwork_2.pth'))
best_model_4.eval()

with open('losses_per_fold_4.pkl', 'rb') as f:
    losses_per_fold_4 = pickle.load(f)

# Concatenate x2, y2, and p2 into a single tensor along the second dimension (i.e., column-wise)

test_dataset = CustomDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            ids, x, y, p, ce = batch
            predicted_ce = model(ids, x, y, p)
            predictions.append(predicted_ce)
            actuals.append(ce)
    
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
    
    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Absolute Error on Test Set: {mae.item():.4f}')
    
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)
    
    return mse, mae,  comparison


print('\nEvaluating model on Test Dataset')
mse_4, mae_4, comparison_4 = evaluate_model(best_model_4, test_loader)


end_time_4 = time.time()
time_diff_4 = abs(start_time_4 - end_time_4)/60

print("NN with ID")
print(f"Time taken (minutes): {time_diff_4:.2f}") #7.86
print(f"Losses per fold: {losses_per_fold_4}") #[16.06855511242461, 15.251242824176765, 15.542200811193624, 15.84541655906479, 15.695355646398818]
print(f"MSE {mse_4}")   #15.26869010925293


predicted_ce_4 = comparison_4[:, 0]
actual_ce_4 = comparison_4[:, 1]

plt.scatter(actual_ce, predicted_ce_4, alpha=0.5, edgecolor='white')
plt.xlabel('Actual CE')
plt.ylabel('Predicted CE')
plt.title('Predicted vs Actual Certainty Equivalents')
plt.plot([min(actual_ce), max(actual_ce)], [min(actual_ce), max(actual_ce)], color='red', linestyle='--')  # y=x line
plt.grid(True)
plt.show()

residuals = predicted_ce_3 - actual_ce
plt.hist(residuals, bins=30, alpha=0.7, color='skyblue')
plt.xlabel('Residual (Predicted CE - Actual CE)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)
plt.show()

epochs = range(1, len(losses_per_fold_4) * 10, 10)
plt.plot(epochs, losses_per_fold_4, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()



###############################################################################################################################


# %%
# Model 5: Utility and Inverse
    
#IT WORKED!



start_time_5 = time.time()

class Utility(nn.Module):
    def __init__(self):
        super(Utility, self).__init__()
    
    # Initialize lambda as a learnable parameter with a default value (e.g., 2.25)
        #self.lambd = nn.Parameter(torch.tensor(2.25, dtype=torch.float32, requires_grad=True))
        
        self.probability_weighting = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
        self.utility = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
        self.inverse = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
    def forward(self, x, y, p):
        u_x = self.utility(x)
        u_y = self.utility(y)
        pi_p = self.probability_weighting(p)
        u_ce = pi_p *u_x + (1 - pi_p) * u_y
        ce = self.inverse(u_ce)
        return ce
        

    
patience = 10
early_stop = False

criterion = nn.MSELoss()
model_5 = Utility()
optimizer = torch.optim.Adam(model_5.parameters(), lr=0.01)
batch_size = 8


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0
        for batch in train_loader:
            x, y, p, ce = batch 
            
            optimizer.zero_grad()
            predicted_ce = model(x, y, p)
            
            primary_loss = criterion(predicted_ce, ce)
            
            u_x = model.utility(x)
            x_reconstructed = model.inverse(u_x)
            inverse_loss_x = criterion(x_reconstructed, x)
            
            u_y = model.utility(y)
            y_reconstructed = model.inverse(u_y)
            inverse_loss_y = criterion(y_reconstructed, y)
            
            u_ce = model.utility(ce)
            ce_reconstructed = model.inverse(u_ce)
            inverse_loss_ce = criterion(ce_reconstructed, ce)
            
            inverse_consistency_loss = (inverse_loss_x + inverse_loss_y + 2*inverse_loss_ce)/4
            
            alpha = 0.2
            loss = primary_loss + alpha * inverse_consistency_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y, p, ce = batch
                predicted_ce = model(x, y, p)
                
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break
    return best_model_state, best_val_loss
        
        
        
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    
    with torch.no_grad():  # No gradient updates needed for evaluation
        for batch in test_loader:
            x, y, p, ce = batch
            predicted_ce = model(x, y, p)
            predictions.append(predicted_ce)
            actuals.append(ce)
    
    # Convert lists to tensors
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    # Calculate MSE between predicted and actual CE values
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
    
    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mae.item():.4f}')
    
    # Print the comparison
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)
    
    return mse, mae, comparison


# Create a TensorDataset
dataset = TensorDataset(x,y,p,ce)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_5 = []

best_model_state_5 = None
best_fold_val_loss = float('inf')

#"""
#train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_5 = Utility()
    optimizer = torch.optim.Adam(model_5.parameters(), lr=0.01)
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_5, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_5.append(fold_val_loss) 

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_5 = fold_best_model_state


# Load the best model for evaluation
best_model_5 = Utility()
best_model_5.load_state_dict(best_model_state_5)
# Load the best model for evaluation
best_model_5 = Utility()
best_model_5.load_state_dict(best_model_state_5)

torch.save(best_model_state_5, 'Utility.pth')

with open('losses_per_fold_5.pkl', 'wb') as f:
    pickle.dump(losses_per_fold_5, f)

#"""

best_model_5 = Utility()
best_model_5.load_state_dict(torch.load('Utility.pth'))
best_model_5.eval()

with open('losses_per_fold_5.pkl', 'rb') as f:
    losses_per_fold_5 = pickle.load(f)


test_dataset = TensorDataset(x2,y2,p2,ce2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('\nEvaluating model on Test Dataset')
mse_5, mae_5, comparison_5 = evaluate_model(best_model_5, test_loader)


end_time_5 = time.time()
time_diff_5 = abs(start_time_5 - end_time_5)/60

print("Utility")
print(f"Time taken (minutes): {time_diff_5:.2f}") #7.86
print(f"Losses per fold: {losses_per_fold_5}") #[16.06855511242461, 15.251242824176765, 15.542200811193624, 15.84541655906479, 15.695355646398818]
print(f"MSE {mse_5}")   #15.7990




def graph_prob(model, function, name, name2):
    p_values = torch.linspace(0, 1, 1000).unsqueeze(1)
    model.eval()
    with torch.no_grad():  # We do not need to track gradients here
        pi_p = function(p_values)
        
    p_values = p_values.numpy().flatten()
    pi_p = pi_p.numpy().flatten()
    
    data = pd.DataFrame({'p (Probability)': p_values, name: pi_p})
    
    plt.figure(figsize=(8, 6))
    
    # Adding the 45-degree line with a subtle dashed linestyle
    plt.plot(p_values, p_values, linestyle='--', color='black', label='45° Line (π(p) = p)', linewidth=1)
    
    # Adding vertical lines at x = 1/8 and x = 7/8 with subtle dotted linestyles
    #plt.axvline(x=1/8, linestyle=':', color='black', label='p = 1/8', linewidth=1)
    #plt.axvline(x=7/8, linestyle='-.', color='black', label='p = 7/8', linewidth=1)
    
    tick_values = [i/8 for i in range(9)]  # 0, 1/8, 2/8, ..., 1
    tick_labels = [f'{i/8}' for i in range(9)]  # '0/8', '1/8', ..., '8/8'
    plt.xticks(ticks=tick_values, labels=tick_labels)
    
    plt.yticks(ticks=tick_values, labels=tick_labels)
    
    # Plotting the learned probability weighting function with a distinct dark green color
    sns.lineplot(x='p (Probability)', y=name, data=data, linewidth=2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Set y-limit slightly above the max value for visibility


    # Setting labels and title
    plt.title(name, fontsize=14)
    plt.xlabel('Probability p', fontsize=12)
    plt.ylabel(name, fontsize=12)

    # Removing the grid for a cleaner look
    plt.legend(fontsize=10)
    plt.grid()
    plt.tight_layout()
    
    filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
    plt.savefig(filename, format='pdf')
    
    plt.show()




graph_prob(best_model_5, best_model_5.probability_weighting, 'Probability Weighting π(p)', name2 = 'Model 5')


name2 = "Model 5"
name = "Utility"
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_5.eval()
with torch.no_grad():
    ut_x = -model_5.utility(x_values)

x_values = x_values.numpy().flatten()
ut_x = ut_x.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'u': ut_x})

plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='u', data=data, label='Utility u(x)')

# Create the custom line with slope corresponding to 63.43 degrees for negative x-values and 1 for non-negative
line_x = np.linspace(-30, 30, 1000)
line_y = np.piecewise(line_x, 
                      [line_x < 0, line_x >= 0], 
                      [lambda x: 2 * x,  # Line with slope 2 for negative x
                       lambda x: x])  # 45° line for non-negative x

plt.plot(line_x, line_y, linestyle='--', color='red', label='Piecewise linear with λ = 2')  # Custom line

plt.title("Utility")
plt.xlabel('x', fontsize = 12)
plt.xlim(-30, 30)
plt.ylim(-65, 30)
plt.legend(fontsize=10)
plt.ylabel("Utility u(x)", fontsize = 12)
plt.grid(True)
plt.legend()
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()



name = "Inverse Utility"
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_5.eval()
with torch.no_grad():  # We do not need to track gradients here
    vt_x = model_5.inverse(x_values)
    vt_x = -vt_x
    
x_values = x_values.numpy().flatten()
vt_x = vt_x.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'v': vt_x})

plt.figure(figsize=(8, 6))
sns.lineplot(x = 'x',  y='v', data = data)
plt.title("Inverse Utility")
plt.xlabel('x')
plt.ylabel("Inverse Utility")
line_x = np.linspace(-30, 30, 1000)
line_y = np.piecewise(line_x, 
                      [line_x < 0, line_x >= 0], 
                      [lambda x: x,  # Line with slope 2 for negative x
                       lambda x: x/2])  # 45° line for non-negative x

plt.plot(line_x, line_y, linestyle='--', color='red', label='Piecewise linear with λ = 2')  # Custom line
plt.xlim(-30, 30)
plt.grid(True)
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()



name = 'Composition'
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_5.eval()
with torch.no_grad():
    z = model_5.inverse(model_5.utility(x_values))

x_values = x_values.numpy().flatten()
z = z.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'z': z})

plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='z', data=data, label='Composition')
#plt.plot(x_values, x_values, linestyle='--', color='red', label='45° Line (y=x)')  # 45-degree line
plt.title("Composition")
plt.xlabel('x')
plt.xlim(-30, 30)
plt.ylabel("Composition")
plt.plot(x_values, x_values, linestyle='--', color='red', label=r'45° Line $(u^{-1}(u(x)) = x)$', linewidth=1)
plt.grid(True)
plt.legend()
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()




#x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
#with torch.no_grad():
#    z = model_5.utility(model_5.inverse(x_values))
    
#print(z)


####################################################################################################

 #%%

# Model 8 CPT with Inverse



start_time_8 = time.time()

class CPT_Utility(nn.Module):
    def __init__(self):
        super(CPT_Utility, self).__init__()
    
    # Initialize lambda as a learnable parameter with a default value (e.g., 2.25)
        #self.lambd = nn.Parameter(torch.tensor(2.25, dtype=torch.float32, requires_grad=True))
        
        self.probability_weighting_gains = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.probability_weighting_losses = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
        self.utility = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
        self.inverse = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        
    def forward(self, x, y, p):
        u_x = self.utility(x)
        u_y = self.utility(y)
        pi_p_x = torch.where(
            (x > 0) & (y >= 0),  # Gains for both outcomes
            self.probability_weighting_gains(p), 
            torch.where(
                (x < 0) & (y <= 0),  # Losses for both outcomes
                self.probability_weighting_losses(p), 
                self.probability_weighting_gains(p)  # Mixed outcomes
            )
        )
        pi_p_y = torch.where(
            (x > 0) & (y >= 0),  # Gains for both
            1 - pi_p_x,
            torch.where(
                (x < 0) & (y <= 0),  # Losses for both
                1 - pi_p_x,
                self.probability_weighting_losses(1 - p)  # Mixed outcomes
            )
        )
        
        u_ce = pi_p_x *u_x + pi_p_y * u_y
        ce = self.inverse(u_ce)
        return ce
        
        
    
patience = 10
early_stop = False

criterion = nn.MSELoss()
model_8 = CPT_Utility()
optimizer = torch.optim.Adam(model_8.parameters(), lr=0.01)
batch_size = 8


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0
        for batch in train_loader:
            x, y, p, ce = batch 
            
            optimizer.zero_grad()
            predicted_ce = model(x, y, p)
            
            primary_loss = criterion(predicted_ce, ce)
            
            u_x = model.utility(x)
            x_reconstructed = model.inverse(u_x)
            inverse_loss_x = criterion(x_reconstructed, x)
            
            u_y = model.utility(y)
            y_reconstructed = model.inverse(u_y)
            inverse_loss_y = criterion(y_reconstructed, y)
            
            u_ce = model.utility(ce)
            ce_reconstructed = model.inverse(u_ce)
            inverse_loss_ce = criterion(ce_reconstructed, ce)
            
            inverse_consistency_loss = (inverse_loss_x + inverse_loss_y +  inverse_loss_ce)/3
            
            alpha = 0.2
            loss = primary_loss + alpha * inverse_consistency_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y, p, ce = batch
                predicted_ce = model(x, y, p)
                
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break
    return best_model_state, best_val_loss
        
        
        


#"""
# Create a TensorDataset
dataset = TensorDataset(x,y,p,ce)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_8 = []

best_model_state_8 = None
best_fold_val_loss = float('inf')

#train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_8 = CPT_Utility()
    optimizer = torch.optim.Adam(model_8.parameters(), lr=0.01)
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_8, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_8.append(fold_val_loss) 

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_8 = fold_best_model_state


torch.save(best_model_state_8, 'CPT_Utility.pth')

with open('losses_per_fold_8.pkl', 'wb') as f:
    pickle.dump(losses_per_fold_8, f)




#"""
# Load the best model for evaluation
best_model_8 = CPT_Utility()
best_model_8.load_state_dict(torch.load('CPT_Utility.pth'))
best_model_8.eval()

with open('losses_per_fold_8.pkl', 'rb') as f:
    losses_per_fold_8 = pickle.load(f)


test_dataset = TensorDataset(x2,y2,p2,ce2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    
    with torch.no_grad():  # No gradient updates needed for evaluation
        for batch in test_loader:
            x, y, p, ce = batch
            predicted_ce = model(x, y, p)
            predictions.append(predicted_ce)
            actuals.append(ce)
    
    # Convert lists to tensors
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    # Calculate MSE between predicted and actual CE values
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
    
    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Absolute Error on Test Set: {mae.item():.4f}')
    
    # Print the comparison
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)
    
    return mse, mae, comparison

print('\nEvaluating model on Test Dataset')
mse_8, mae_8, comparison_8 = evaluate_model(best_model_8, test_loader)


end_time_8 = time.time()
time_diff_8 = abs(start_time_8 - end_time_8)/60

print("CPT Utility")
print(f"Time taken (minutes): {time_diff_8:.2f}") #7.86
print(f"Losses per fold: {losses_per_fold_8}") #[16.06855511242461, 15.251242824176765, 15.542200811193624, 15.84541655906479, 15.695355646398818]
print(f"MSE {mse_8}")   #15.7990





graph_prob2(best_model_8, best_model_8.probability_weighting_gains, best_model_8.probability_weighting_losses, 'Probability Weighting Gains π(p)', 'Probability Weighting Losses', 'Model 8')




name2 = "Model 8"
name = "Utility"
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_8.eval()
with torch.no_grad():
    ut_x = model_8.utility(x_values)
    ut_x = -ut_x

x_values = x_values.numpy().flatten()
ut_x = ut_x.numpy().flatten()

data = pd.DataFrame({'x': -x_values, 'u': ut_x})

plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='u', data=data, label='Utility u(x)')

# Create the custom line with slope corresponding to 63.43 degrees for negative x-values and 1 for non-negative
line_x = np.linspace(-30, 30, 1000)
line_y = np.piecewise(line_x, 
                      [line_x < 0, line_x >= 0], 
                      [lambda x: 2 * x,  # Line with slope 2 for negative x
                       lambda x: x])  # 45° line for non-negative x

plt.plot(line_x, line_y, linestyle='--', color='red', label='Piecewise linear with λ = 2')  # Custom line

plt.title("Utility")
plt.xlabel('x', fontsize = 12)
plt.xlim(-30, 30)
plt.ylim(-65, 30)
plt.legend(fontsize=10)
plt.ylabel("Utility u(x)", fontsize = 12)
plt.grid(True)
plt.legend()
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()



name = "Inverse Utility"
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_8.eval()
with torch.no_grad():  # We do not need to track gradients here
    vt_x = model_8.inverse(x_values)
    vt_x = -vt_x
    
x_values = x_values.numpy().flatten()
vt_x = vt_x.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'v': vt_x})

plt.figure(figsize=(8, 6))
sns.lineplot(x = 'x',  y='v', data = data)
plt.title("Inverse Utility")
plt.xlabel('x')
plt.ylabel("Inverse Utility")
line_x = np.linspace(-30, 30, 1000)
line_y = np.piecewise(line_x, 
                      [line_x < 0, line_x >= 0], 
                      [lambda x: x,  # Line with slope 2 for negative x
                       lambda x: x/2])  # 45° line for non-negative x

plt.plot(line_x, line_y, linestyle='--', color='red', label='Piecewise linear with λ = 2')  # Custom line
plt.xlim(-30, 30)
plt.grid(True)
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()



name = 'Composition'
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_8.eval()
with torch.no_grad():
    z = model_8.inverse(model_8.utility(x_values))

x_values = x_values.numpy().flatten()
z = z.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'z': z})

plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='z', data=data, label='Composition')
#plt.plot(x_values, x_values, linestyle='--', color='red', label='45° Line (y=x)')  # 45-degree line
plt.title("Composition")
plt.xlabel('x')
plt.xlim(-30, 30)
plt.ylabel("Composition")
plt.plot(x_values, x_values, linestyle='--', color='red', label=r'45° Line $(u^{-1}(u(x)) = x)$', linewidth=1)
plt.grid(True)
plt.legend()
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()






predicted_ce = comparison_8[:, 0]
actual_ce = comparison_8[:, 1]

plt.scatter(actual_ce, predicted_ce, alpha=0.5)
plt.xlabel('Actual CE')
plt.ylabel('Predicted CE')
plt.title('Predicted vs Actual Certainty Equivalents')
plt.plot([min(actual_ce), max(actual_ce)], [min(actual_ce), max(actual_ce)], color='red', linestyle='--')  # y=x line
plt.grid(True)
plt.show()

epochs = range(1, len(losses_per_fold_8) * 10, 10)
plt.plot(epochs, losses_per_fold_8, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


name = 'Distribution of Residuals' 
residuals = predicted_ce - actual_ce
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residual (Predicted CE - Actual CE)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()





##################################################################################################


#Model 9: Linear PT with Individual Identification
    
    
    
    



start_time_9 = time.time()




class ProspectTheoryWithID(nn.Module):
    def __init__(self):
        super(ProspectTheoryWithID, self).__init__()
        
        # Embedding layer for ID data
        self.embedding = nn.Embedding(num_embeddings=2940, embedding_dim=32)  # Embedding for 2939 unique IDs
        
        # Probability weighting function (π(p))
        self.probability_weighting = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # Layers to predict CE from PT and the embedding 
        self.fc1 = nn.Linear(32 + 1, 64)  # 32 for ID embedding, 1 for PT
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, ids, x, y, p):
        # Get embedding for IDs
        embedded_ids = self.embedding(ids)  # Shape: (batch_size, embedding_dim)

        # Calculate the probability weighting
        pi_p = self.probability_weighting(p)  # Shape: (batch_size, 1)

        # Calculate PT using π(p) * x + (1 - π(p)) * y
        pt = pi_p * x + (1 - pi_p) * y  # Shape: (batch_size, 1)

        # Concatenate PT with ID embedding
        combined_input = torch.cat((embedded_ids, pt), dim=1)  # Shape: (batch_size, 32 + 1)

        # Forward pass through fully connected layers
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        ce = self.fc3(x)

        return ce

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, df):
        self.ids = torch.tensor(df['subject_global'].values, dtype=torch.long)  # Convert IDs to integers
        self.x = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)
        self.p = torch.tensor(df['p'].values, dtype=torch.float32).unsqueeze(1)
        self.ce = torch.tensor(df['ce'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.x[idx], self.y[idx], self.p[idx], self.ce[idx]


# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids, x, y, p, ce = batch
            
            optimizer.zero_grad()
            predicted_ce = model(ids, x, y, p)
            loss = criterion(predicted_ce, ce)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                ids, x, y, p, ce = batch
                predicted_ce = model(ids, x, y, p)
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            #print(f"Getting better! Epoch {epoch}")
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break
        print(epoch)
    return best_model_state, best_val_loss


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            ids, x, y, p, ce = batch
            predicted_ce = model(ids, x, y, p)
            predictions.append(predicted_ce)
            actuals.append(ce)
    
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
    
    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Absolute Error on Test Set: {mae.item():.4f}')
    
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)
    
    return mse, mae,  comparison

patience = 10
early_stop = False
criterion = nn.MSELoss()
model_9 = ProspectTheoryWithID()
optimizer = torch.optim.Adam(model_9.parameters(), lr=0.01)
batch_size = 8


train_dataset = CustomDataset(train_df)
dataset = train_dataset

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_9 = []

best_model_state_9 = None
best_fold_val_loss = float('inf')


#"""
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_9 = ProspectTheoryWithID()
    optimizer = torch.optim.Adam(model_9.parameters(), lr=0.01)
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_9, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_9.append(fold_val_loss) 

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_9 = fold_best_model_state


torch.save(best_model_state_9, 'ProspectTheoryWithID.pth')

with open('losses_per_fold_9.pkl', 'wb') as f:
    pickle.dump(losses_per_fold_9, f)

#"""

# Load the best model for evaluation
best_model_9 = ProspectTheoryWithID()
best_model_9.load_state_dict(torch.load('ProspectTheoryWithID.pth'))
best_model_9.eval()


with open('losses_per_fold_9.pkl', 'rb') as f:
    losses_per_fold_9 = pickle.load(f)


# Concatenate x2, y2, and p2 into a single tensor along the second dimension (i.e., column-wise)

test_dataset = CustomDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('\nEvaluating model on Test Dataset')
mse_9, mae_9, comparison_9 = evaluate_model(best_model_9, test_loader)


end_time_9 = time.time()
time_diff_9 = abs(start_time_9 - end_time_9)/60

print("ProspectTheoryWithID")
print(f"Time taken (minutes): {time_diff_9:.2f}") #7.86
print(f"Losses per fold: {losses_per_fold_9}") #[16.06855511242461, 15.251242824176765, 15.542200811193624, 15.84541655906479, 15.695355646398818]
print(f"MSE {mse_9}")   #15.26869010925293


predicted_ce_9 = comparison_9[:, 0]
actual_ce_9 = comparison_9[:, 1]










p_values = torch.linspace(0, 1, 1000).unsqueeze(1)

name2 = 'Model 9'
name = "Probability"
model_9.eval()
#graph_prob(best_model_9, best_model_9.probability_weighting,'Probability Weighting π(p)', name2 = name2)


with torch.no_grad():
    pi_p = model_9.probability_weighting(p_values)

    
p_values = p_values.numpy().flatten()
pi_p = pi_p.numpy().flatten()

data = pd.DataFrame({'p (Probability)': p_values, name: pi_p})

plt.figure(figsize=(8, 6))

# Adding the 45-degree line with a subtle dashed linestyle
plt.plot(p_values, p_values, linestyle='--', color='black', label='45° Line (π(p) = p)', linewidth=1)

# Adding vertical lines at x = 1/8 and x = 7/8 with subtle dotted linestyles
#plt.axvline(x=1/8, linestyle=':', color='black', label='p = 1/8', linewidth=1)
#plt.axvline(x=7/8, linestyle='-.', color='black', label='p = 7/8', linewidth=1)

#tick_values = [i/8 for i in range(9)]  # 0, 1/8, 2/8, ..., 1
#tick_labels = [f'{i/8}' for i in range(9)]  # '0/8', '1/8', ..., '8/8'
#plt.xticks(ticks=tick_values, labels=tick_labels)

#plt.yticks(ticks=tick_values, labels=tick_labels)

# Plotting the learned probability weighting function with a distinct dark green color
sns.lineplot(x='p (Probability)', y=name, data=data, linewidth=2)

#plt.xlim(0, 1)
#plt.ylim(0, 1)  # Set y-limit slightly above the max value for visibility


# Setting labels and title
plt.title(name, fontsize=14)
plt.xlabel('Probability p', fontsize=12)
plt.ylabel(name, fontsize=12)

# Removing the grid for a cleaner look
plt.legend(fontsize=10)
plt.grid()
plt.tight_layout()

filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')

plt.show()
























plt.scatter(actual_ce, predicted_ce_9, alpha=0.5, edgecolor='white')
plt.xlabel('Actual CE')
plt.ylabel('Predicted CE')
plt.title('Predicted vs Actual Certainty Equivalents')
plt.plot([min(actual_ce), max(actual_ce)], [min(actual_ce), max(actual_ce)], color='red', linestyle='--')  # y=x line
plt.grid(True)
plt.show()

residuals = predicted_ce_9 - actual_ce
plt.hist(residuals, bins=30, alpha=0.7, color='skyblue')
plt.xlabel('Residual (Predicted CE - Actual CE)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)
plt.show()

epochs = range(1, len(losses_per_fold_9) * 10, 10)
plt.plot(epochs, losses_per_fold_9, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()



#########################################

print(f"ProspectTheoryWithID: {mse_9.item()}")
print(f"CPT_Utility: {mse_8.item()}")
print(f"One utility and one weighting probability: {mse_5.item()}")
print(f"NN with (x,p,y,id): {mse_4.item()}")
print(f"NN with (x,p;y): {mse_3.item()}")
print(f"Neural-CPT (Linear): {mse_2.item()}")
print(f"Neural-PT (Linear): {mse_1.item()}")



print(f"ProspectTheoryWithID: {mae_9.item()}")
print(f"CPT_Utility: {mae_8.item()}")
print(f"One utility and one weighting probability: {mae_5.item()}")
print(f"NN with ID: {mae_4.item()}")
print(f"NN with (x,p;y):{mae_3.item()}")
print(f"CPT Linear Utility {mae_2.item()}")
print(f"PT Linear Utility{mae_1.item()}")

#################################################################################################


# %%
# Computing Restrictiveness
"""
import numpy as np
import pandas as pd


import pickle

random.seed(42)
num_samples = 80000000

data = np.random.uniform(
    low=[0, 0, 0, 0, 10, 20, 0, 5, 0, 0, 0, 0, 0, 5],
    high=[5, 10, 20, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20],
    size=(num_samples, 14)
)

filtered_data = data[(data[:, 0] <= data[:, 1]) & (data[:, 1] <= data[:, 2]) & (data[:, 2] <= data[:, 3]) & (data[:, 3] <= data[:, 4]) & (data[:, 4] <= data[:, 5]) & \
                     (data[:, 6] <= data[:, 7]) & \
                     (data[:, 6] <= data[:, 8]) & (data[:, 8] <= data[:, 9]) & (data[:, 10] <= data[:, 11]) & (data[:, 11] <= data[:, 12]) & (data[:, 12] <= data[:, 13]) & \
                     (data[:, 10] <= data[:, 3]) & (data[:, 3] <= data[:, 11]) & \
                     (data[:, 7] <= data[:, 13]) & (data[:, 13] <= (data[:, 5]))]


with open('pos_data.pkl', 'wb') as f:
    pickle.dump(filtered_data, f)
    
    
    
num_samples = 85000000

data = np.random.uniform(
    low=[-5, -10, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20],
    high=[0, 0, 0, 0, -5, -10, 0, -5, 0, 0, 0, 0, -5],
    size=(num_samples, 13)
)

filtered_data = data[(data[:, 0] >= data[:, 1]) & (data[:, 1] >= data[:, 2]) & (data[:, 2] >= data[:, 3]) & (data[:, 3] >= data[:, 4]) & \
                     (data[:, 5] >= data[:, 6]) & \
                     (data[:, 5] >= data[:, 7]) & (data[:, 7] >= data[:, 8]) & (data[:, 8] >= data[:, 9]) & (data[:, 10] >= data[:, 11]) & (data[:, 11] >= data[:, 12])]
                     #& \
                     #(data[:, 8] >= data[:, 2]) & (data[:, 2] >= data[:, 9]) & \
                     #(data[:, 6] >= data[:, 3])] & \
                     #(data[:, 6] >= data[:, 12])]

with open('neg_data.pkl', 'wb') as f:
    pickle.dump(filtered_data, f)
    


    
    
with open('pos_data.pkl', 'rb') as f:
    pos_data = pickle.load(f)
    
with open('neg_data.pkl', 'rb') as f:
    neg_data = pickle.load(f)
    

z = pd.DataFrame(pos_data[0:2939,:])
zz = pd.DataFrame(neg_data[0:2939,:])
zzz = pd.DataFrame(np.random.uniform(-20, 0, 2939))

df_combined = pd.concat([z, zz], axis=1)
df_combined = pd.concat([df_combined, zzz], axis=1)

#b = df_combined.head()
#u = b.values
#u = u.flatten()

df_combined = df_combined.values
df_combined = df_combined.flatten()

df_combined = pd.DataFrame(df_combined, columns = ['ce'])
df_combined['x'] = [5, 10, 20, 30, 30, 30, 20, 20, 20, 20, 20, 20, 20, 20, -5, -10, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, 20] * 2939
df_combined['y'] = [0, 0, 0, 0, 10, 20, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0, 0, -5, -10, 0, -5, 0, 0, 0, 0, 0, -5, 'yo']*2939
df_combined['p'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.125, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 0.875, 0.5, 0.5, 0.5, 0.5, 0.5, 0.125, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 0.875, 0.5]*2939 
df_combined['yes'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'yes']*2939
df_combined.loc[df_combined['yes'] == 'yes', 'y'] = df_combined['ce'] 
df_combined.loc[df_combined['yes'] == 'yes', 'ce'] = 0

df_combined['eut'] = df_combined['p']*df_combined['x'] + (1- df_combined['p'])*df_combined['y'] 
df_combined['d_eut'] = (df_combined['eut']-df_combined['ce'])**2

observations = [i for i in range(2939) for _ in range(28)]
df_combined['subject_global'] = observations

repetitions = [i for i in range(1, 29)] * 2939

df_combined['equiv_nr']  = repetitions

df_combined = df_combined[['subject_global', 'x', 'y', 'p', 'ce', 'equiv_nr', 'eut', 'd_eut']]

with open('res.pkl', 'wb') as f:
    pickle.dump(df_combined, f)
"""
##################################################################################################################


# %% Model 11:

"""

start_time_11 = time.time()



class CPT_Individual(nn.Module):
    def __init__(self):
        super(CPT_Individual, self).__init__()
        # Embedding layer for the ID data
        self.embedding = nn.Embedding(num_embeddings=2940, embedding_dim=32)  # 2939 unique IDs, embedding size of 32

        self.probability_weighting_gains = nn.Sequential(
            nn.Linear(32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )
        
        self.probability_weighting_losses = nn.Sequential(
            nn.Linear(32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )
        
        self.utility = nn.Sequential(
            nn.Linear(32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )
        
        self.inverse = nn.Sequential(
            nn.Linear(32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

    def forward(self, x, y, p, ide):
        embedded_ids = self.embedding(ide)
        
        
        if x.dim() == 3:
            # If x is 3D (batch_size, seq_len, 1), reshape to (batch_size, 1)
            x = x.view(x.size(0), -1)
            y = y.view(y.size(0), -1)
            p = p.view(p.size(0), -1)
        elif x.dim() == 2:
            # If x is already 2D (batch_size, 1), keep it as is
            x = x.view(-1, 1)
            y = y.view(-1, 1)
            p = p.view(-1, 1)
        
        p_compl = 1 - p
        
        input_x = torch.cat((embedded_ids, x), dim=1)
        input_y = torch.cat((embedded_ids, y), dim=1)
        input_p = torch.cat((embedded_ids, p), dim=1)
        input_p_compl = torch.cat((embedded_ids, p_compl), dim=1)
        
        u_x = self.utility(input_x)
        u_y = self.utility(input_y)
        
        gains_mask = (x > 0) & (y >= 0)
        losses_mask = (x < 0) & (y <= 0)
        mixed_mask = ~(gains_mask | losses_mask)  # For mixed outcomes

        # Initialize pi_p_x and pi_p_y as zero tensors
        pi_p_x = torch.zeros_like(x)
        pi_p_y = torch.zeros_like(y)

        # Apply probability weighting functions conditionally using masks
        pi_p_x[gains_mask] = self.probability_weighting_gains(input_p[gains_mask])
        pi_p_y[gains_mask] = 1 - pi_p_x[gains_mask]

        pi_p_x[losses_mask] = self.probability_weighting_losses(input_p[losses_mask])
        pi_p_y[losses_mask] = 1 - pi_p_x[losses_mask]

        pi_p_x[mixed_mask] = self.probability_weighting_gains(input_p[mixed_mask])
        pi_p_y[mixed_mask] = self.probability_weighting_losses(input_p_compl[mixed_mask])

        
        
        u_ce = pi_p_x * u_x + pi_p_y * u_y
        
        input_u_ce = torch.cat((embedded_ids, u_ce), dim=1)
        ce = self.inverse(input_u_ce)
        
        return ce



        
        
    
patience = 10
early_stop = False

criterion = nn.MSELoss()
model_11 = CPT_Individual()
optimizer = torch.optim.Adam(model_11.parameters(), lr=0.01)
batch_size = 8


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    best_model_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x, y, p, ide, ce = batch

            optimizer.zero_grad()
            # Forward pass through the model
            predicted_ce = model(x, y, p, ide)

            # Primary loss
            primary_loss = criterion(predicted_ce, ce)

            # Consistency loss calculations
            embedded_ids = model.embedding(ide)

            # Utility consistency for x
            input_x = torch.cat((embedded_ids, x), dim=1)
            u_x = model.utility(input_x)
            input_u_x = torch.cat((embedded_ids, u_x), dim=1)
            x_reconstructed = model.inverse(input_u_x)
            inverse_loss_x = criterion(x_reconstructed, x)

            # Utility consistency for y
            input_y = torch.cat((embedded_ids, y), dim=1)
            u_y = model.utility(input_y)
            input_u_y = torch.cat((embedded_ids, u_y), dim=1)
            y_reconstructed = model.inverse(input_u_y)
            inverse_loss_y = criterion(y_reconstructed, y)

            # Utility consistency for ce
            input_ce = torch.cat((embedded_ids, ce), dim=1)
            u_ce = model.utility(input_ce)
            input_u_ce = torch.cat((embedded_ids, u_ce), dim=1)
            ce_reconstructed = model.inverse(input_u_ce)
            inverse_loss_ce = criterion(ce_reconstructed, ce)

            # Calculate the total inverse consistency loss
            inverse_consistency_loss = (inverse_loss_x + inverse_loss_y + 2 * inverse_loss_ce) / 4

            # Total loss
            alpha = 0.2
            loss = primary_loss + alpha * inverse_consistency_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y, p, ide, ce = batch
                predicted_ce = model(x, y, p, ide)
                loss = criterion(predicted_ce, ce)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Track the best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model state
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Logging for every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

    return best_model_state, best_val_loss
        
        
        



# Create a TensorDataset
dataset = TensorDataset(x,y,p,ide,ce)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

losses_per_fold_11 = []

best_model_state_11 = None
best_fold_val_loss = float('inf')

#train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model_11 = CPT_Individual()
    optimizer = torch.optim.Adam(model_11.parameters(), lr=0.01)

    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    fold_best_model_state, fold_val_loss = train_model(model_11, criterion, optimizer, train_loader, val_loader)
    losses_per_fold_11.append(fold_val_loss)

    # Track the best model across all folds
    if fold_val_loss < best_fold_val_loss:
        best_fold_val_loss = fold_val_loss
        best_model_state_11 = fold_best_model_state


# Load the best model for evaluation
best_model_11 = CPT_Individual()
best_model_11.load_state_dict(best_model_state_11)


test_dataset = TensorDataset(x2,y2,p2,ide2,ce2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []

    with torch.no_grad():  # No gradient updates needed for evaluation
        for batch in test_loader:
            x, y, p, ide, ce = batch
            predicted_ce = model(x, y, p, ide)
            predictions.append(predicted_ce)
            actuals.append(ce)

    # Convert lists to tensors
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)

    # Calculate MSE between predicted and actual CE values
    mse = criterion(predictions, actuals)
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')

    mae = nn.L1Loss()(predictions, actuals)
    print(f'Mean Absolute Error on Test Set: {mae.item():.4f}')

    # Print the comparison
    comparison = torch.cat((predictions, actuals), dim=1).numpy()
    print("Predicted CE | Actual CE")
    print(comparison)

    return mse, mae, comparison

print('\nEvaluating model on Test Dataset')
mse_11, mae_11, comparison_11 = evaluate_model(best_model_11, test_loader)


end_time_11 = time.time()
time_diff_11 = abs(start_time_11 - end_time_11)/60

print("Utility Individual")
print(f"Time taken (minutes): {time_diff_11:.2f}") #7.86
print(f"Losses per fold: {losses_per_fold_11}") #[16.06855511242461, 15.251242824176765, 15.542200811193624, 15.84541655906479, 15.695355646398818]
print(f"MSE {mse_11}")   #15.7990


"""

"""
graph_prob2(model_, model_8.probability_weighting_gains, model_8.probability_weighting_losses, 'Probability Weighting Gains π(p)', 'Probability Weighting Losses', 'Model 8')




name2 = "Model 11"
name = "Utility Individual"
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_8.eval()
with torch.no_grad():
    ut_x = model_8.utility(x_values)
    ut_x = -ut_x

x_values = x_values.numpy().flatten()
ut_x = ut_x.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'u': ut_x})

plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='u', data=data, label='Utility u(x)')

# Create the custom line with slope corresponding to 63.43 degrees for negative x-values and 1 for non-negative
line_x = np.linspace(-30, 30, 1000)
line_y = np.piecewise(line_x, 
                      [line_x < 0, line_x >= 0], 
                      [lambda x: 2 * x,  # Line with slope 2 for negative x
                       lambda x: x])  # 45° line for non-negative x

plt.plot(line_x, line_y, linestyle='--', color='red', label='Piecewise linear with λ = 2')  # Custom line

plt.title("Utility")
plt.xlabel('x', fontsize = 12)
plt.xlim(-30, 30)
plt.ylim(-65, 30)
plt.legend(fontsize=10)
plt.ylabel("Utility u(x)", fontsize = 12)
plt.grid(True)
plt.legend()
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()



name = "Inverse Utility"
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_8.eval()
with torch.no_grad():  # We do not need to track gradients here
    vt_x = model_8.inverse(x_values)
    vt_x = -vt_x
    
x_values = x_values.numpy().flatten()
vt_x = vt_x.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'v': vt_x})

plt.figure(figsize=(8, 6))
sns.lineplot(x = 'x',  y='v', data = data)
plt.title("Inverse Utility")
plt.xlabel('x')
plt.ylabel("Inverse Utility")
line_x = np.linspace(-30, 30, 1000)
line_y = np.piecewise(line_x, 
                      [line_x < 0, line_x >= 0], 
                      [lambda x: x,  # Line with slope 2 for negative x
                       lambda x: x/2])  # 45° line for non-negative x

plt.plot(line_x, line_y, linestyle='--', color='red', label='Piecewise linear with λ = 2')  # Custom line
plt.xlim(-30, 30)
plt.grid(True)
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()



name = 'Composition'
x_values = torch.linspace(-30, 30, 1000).unsqueeze(1)
model_8.eval()
with torch.no_grad():
    z = model_8.inverse(model_8.utility(x_values))

x_values = x_values.numpy().flatten()
z = z.numpy().flatten()

data = pd.DataFrame({'x': x_values, 'z': z})

plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='z', data=data, label='Composition')
#plt.plot(x_values, x_values, linestyle='--', color='red', label='45° Line (y=x)')  # 45-degree line
plt.title("Composition")
plt.xlabel('x')
plt.xlim(-30, 30)
plt.ylabel("Composition")
plt.plot(x_values, x_values, linestyle='--', color='red', label=r'45° Line $(u^{-1}(u(x)) = x)$', linewidth=1)
plt.grid(True)
plt.legend()
filename = f"{name2.replace(' ', '_')}_{name.replace(' ', '_')}.pdf"
plt.savefig(filename, format='pdf')
plt.show()
"""
