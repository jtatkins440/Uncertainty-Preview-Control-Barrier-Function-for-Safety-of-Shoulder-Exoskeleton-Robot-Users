import os
import yaml
import time
import torch
from torch import nn
import numpy as np

''' ### HOW TO USE IT:
1. Give it a model_name parameter like "ts_model" if the model and config file are in the same folder as this script. If not, also pass in a model_dir_path parameter to the model path.
    inf_model = TsModelWrapper(model_name)

2. To do predictions, arrange an input as a numpy array like 
    input = np.array([Velocity, u_max, w_max])
and then pass that input to the model's predict method like:
    Ts_predict = inf_model.predict(input)

That's it! Ask John Atkins for any other help.
'''

def trainLoop(model, dataloader, optimizer):
    size = len(dataloader.dataset) #len(self.dataloader_training.dataset)
    train_loss = 0.0
    
    loss_fn = nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= size
    return

def testLoop(model, dataloader, bverbose = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()

    test_loss /= size

    if bverbose:
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    
    return test_loss

def trainModel(model, train_dataloader, test_dataloader, optimizer, scheduler, epochs = 30, min_epoch_num = 4, loss_rate_max = 0.999, 
        loss_mean_last = 1000, loss_rate_break_counts_max = 4, verbose = True):

    epoch_lists = []
    correct_rate_lists = []
    loss_mean_lists = []
    loss_mse_lists = []


    train_time_start = 0.0
    train_time = 0.0
    train_time_lists = []

    test_time_start = 0.0
    test_time = 0.0
    test_time_lists = []

    loss_rate_break_counts = 0

    for t in range(epochs):
        if verbose: print(f"Epoch {t}\n-------------------------------")

        train_time_start = time.time()
        trainLoop(model, train_dataloader, optimizer)
        train_time = time.time() - train_time_start
        
        test_time_start = time.time()
        loss_mean = testLoop(model, test_dataloader)
        test_time = time.time() - test_time_start

        train_time_lists.append(train_time)
        test_time_lists.append(test_time)
        epoch_lists.append(t)

        loss_mean_lists.append(loss_mean)

        loss_rate = loss_mean / loss_mean_last
        if verbose: print(f"Loss rate: {loss_rate}")
        if verbose: print(f"Training time: {train_time}")
        print(f"Mean loss: {loss_mean}")

        scheduler.step()
        if min_epoch_num < t:
            if loss_rate_max < loss_rate:
                loss_rate_break_counts = loss_rate_break_counts + 1
            else:
                loss_rate_break_counts = 0
            if loss_rate_break_counts_max < loss_rate_break_counts:
                break
        loss_mean_last = loss_mean


    if verbose: print("Done!")
    train_res_dict = {'epochs': epoch_lists, 'loss_means' : loss_mean_lists, 'accuracy_rates' : correct_rate_lists, 
        'training_times' : train_time_lists, 'validation_times' : test_time_lists,
        'loss_mse' : loss_mse_lists}

    return train_res_dict, model
    
class TsDataset(torch.utils.data.Dataset):
    def __init__(self, source_df, input_cols, target_cols):
        self.input_t = torch.tensor(source_df[input_cols].to_numpy())
        self.target_t = torch.tensor(source_df[target_cols].to_numpy())
        self.valid_length = self.input_t.size(0)
    def __len__(self):
        return self.valid_length
    def __getitem__(self, idx):
        return self.input_t[idx,:], self.target_t[idx]

class Linear_Conditioner(nn.Module):
    def __init__(self, input_channels, condition_features, condition_channels, output_channels, projection_features, hidden_features, dropout_alpha = 0.2):
        super(Linear_Conditioner, self).__init__() 

        input_features = len(input_channels)
        output_features = len(output_channels)
        self.drop = nn.Dropout
        self.act = nn.Softplus # activation to use in most of model, gives strictly positive outputs
        
        self.input_proj_layer = nn.Sequential(nn.Linear(input_features, hidden_features),
                                             self.act(),
                                             nn.Linear(hidden_features, projection_features))
        # self.drop()
        
        self.condition_proj_layers = nn.ModuleList()
        for i in range(len(condition_channels)):
            self.condition_proj_layers = nn.Sequential(nn.Linear(condition_features, hidden_features),
                                             self.act(),
                                             nn.Linear(hidden_features, projection_features))

        condition_proj_features = projection_features * len(condition_channels)

        self.predict_layer_0 = nn.Sequential(nn.Linear(projection_features + condition_proj_features, hidden_features),
                                             self.act(),
                                             self.drop())
        self.predict_layer_1 = nn.Sequential(nn.Linear(hidden_features + condition_proj_features, hidden_features),
                                             self.act(),
                                             self.drop())
        self.predict_layer_2 = nn.Sequential(nn.Linear(hidden_features + condition_proj_features, hidden_features),
                                             self.act(),
                                             self.drop())
                                             
        self.output_layer = nn.Linear(hidden_features, output_features)

        self.input_channels = input_channels
        self.condition_channels = condition_channels
    def project_conditioners(self, condition_list):
        proj_list = []
        for i in range(len(condition_list)):
            proj_list.append(self.condition_proj_layers(condition_list[i]))
        return torch.cat(proj_list, dim=-1)

    def predict_output(self, proj_input, proj_cond):
        out_h = self.predict_layer_0(torch.cat([proj_input, proj_cond], dim=-1))
        out_h = self.predict_layer_1(torch.cat([out_h, proj_cond], dim=-1)) + out_h
        out_h = self.predict_layer_2(torch.cat([out_h, proj_cond], dim=-1)) + out_h
        return self.output_layer(out_h)
        
    def forward(self, input):
        used_inputs = input[:, self.input_channels]
        condition_inputs = [input[:, self.condition_channels[i]:self.condition_channels[i]+1] for i in range(len(self.condition_channels))] # list of conditioned outputs
        proj_conds = self.project_conditioners(condition_inputs)

        proj_inputs = self.input_proj_layer(used_inputs)
        return self.predict_output(proj_inputs, proj_conds)


class TsModelWrapper():
    def __init__(self, model_name, model_dir_path = None, weights_extension = ".pt", config_extension = "_config.yaml", compile_model = True, use_floats = True):
        self.use_floats = use_floats
        
        if model_dir_path is not None:
            model_path = os.path.join(model_dir_path, model_name + weights_extension)
            config_path = os.path.join(model_dir_path, model_name + config_extension)
        else:
            model_path = model_name + weights_extension
            config_path = model_name + config_extension
            
        with open(config_path) as file:
            config_dict = yaml.safe_load(file)

        base_model = Linear_Conditioner(config_dict["input_channels"],
                                        config_dict["condition_features"],
                                        config_dict["condition_channels"],
                                        config_dict["output_channels"],
                                        config_dict["projection_features"],
                                        config_dict["hidden_features"],
                                        config_dict["dropout_alpha"])

        if self.use_floats:
            base_model = base_model.float()
        else:
            base_model = base_model.double()
            
        base_model.load_state_dict(torch.load(model_path))

        temp_input = np.zeros(shape=(len(config_dict["input_channels"]) + len(config_dict["condition_channels"]) * config_dict["condition_features"]))
        if compile_model:
            self.base_model = torch.compile(base_model.eval())
            temp_output = self.predict(temp_input)
            print("Compiled model!")
        else:
            self.base_model = base_model
            print("Did not compile model!")

        avg_num = 1000
        time_array = np.zeros(shape=(avg_num))
        for i in range(avg_num):
            start_time = time.time()
            temp_output = self.predict(temp_input)
            time_array[i] = time.time() - start_time
        print(f"Inference time: {time_array.mean()} +/- {time_array.std()}s (May vary depending on background processes)")
        loss = config_dict["eval_loss"]
        print(f"Expected stopping time prediction error: {loss * 1000}ms")

    def preprocess_input(self, input):
        if self.use_floats:
                input_t = torch.tensor(input, dtype=torch.float)
        else:
            input_t = torch.tensor(input, dtype=torch.double)
        if len(input_t.size()) == 1:
            input_t = torch.unsqueeze(input_t, 0) # add batch dim
        return input_t
    def postprocess_output(self, output_t):
        return output_t.numpy(force=True)
            
    def predict(self, input):
        input_t = self.preprocess_input(input)
        output_t = self.base_model(input_t)
        return self.postprocess_output(output_t)