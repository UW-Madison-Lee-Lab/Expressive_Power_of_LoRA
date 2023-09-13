from load_model import *
from helper import *
from copy import deepcopy
from tqdm import tqdm
from torch import optim

class approx_fnn:
    def __init__(
        self,
        width,
        target_depth,
        frozen_depth,
        rank,
        use_bias,
        activation,
        std = .25,
    ):
        set_seed()
        
        self.width = width
        self.target_depth = target_depth
        self.frozen_depth = frozen_depth
        self.rank = rank
        self.use_bias = use_bias
        
        # initialize the target model
        self.target_m = FNN(
            depth = target_depth,
            width = width,
            rank = rank,
            std = std,
            use_bias = use_bias,
            apply_lora = False,
            activation = activation,
        )
        
        # initialize the frozen model
        self.frozen_m = FNN(
            depth = frozen_depth,
            width = width,
            rank = rank,
            std = std,
            use_bias = use_bias,
            apply_lora = True,
            activation = activation,
        )
        
        self.criterion = nn.MSELoss()

    def adapt_fnn_sgd(
        self,
        n_epochs,
        batch_size,
        lr,
    ):
        adapted_m = deepcopy(self.frozen_m)
        
        # specify the lora adapter as the parameters to be optimized
        params = []
        for l in range(self.frozen_depth):
            params.append({'params': adapted_m.loralist[l].lora_A, 'lr': lr})
            params.append({'params': adapted_m.loralist[l].lora_B, 'lr': lr})
            
        opt = optim.Adam(params)
            
        for i in tqdm(range(n_epochs)):
            # generate random input from some Gaussian distribution
            x_train = torch.randn(batch_size, self.width) 
            y_train = self.target_m(x_train).detach()
            y_train.requires_grad = False
            
            y_pred = adapted_m(x_train)
            loss = self.criterion(y_pred, y_train)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        return adapted_m
    
    def adapt_fnn_ours(
        self,
        batch_size,
    ):
        tdl = self.frozen_depth // self.target_depth
        adapted_m = deepcopy(self.frozen_m)
        
        if self.use_bias:
            # generate random input from some Gaussian distribution
            z = torch.randn(batch_size, self.width) 
        
        lora_adapter = {}
        for i in range(self.target_depth):
            # use range(l1, l2) layers in the adapted model to approximate the ith layer in the target model 
            l1 = i * tdl
            l2 = (i + 1) * tdl if i < self.target_depth - 1 else self.frozen_depth
            
            # compute the product of the frozen matrices
            frozen_prod_weight = torch.eye(self.width)
            frozen_prod_weight_l2L = {(l2-1): frozen_prod_weight}
            for l in range(l1+1, l2)[::-1]:
                frozen_prod_weight = frozen_prod_weight @ adapted_m.linearlist[l].weight.data
                frozen_prod_weight_l2L[l-1] = frozen_prod_weight
                
            # get the target weight
            target_weight = self.target_m.linearlist[i].weight.data
            
            # compute the discrepancy matrix
            discrepancy_weight = target_weight - frozen_prod_weight
            
            # perform SVD on the discrepancy matrix
            U, S, V = torch.svd(discrepancy_weight)
            # Convert S to a diagonal matrix D
            D = torch.zeros_like(discrepancy_weight)
            D.diagonal().copy_(S)
            
            # compute the lora adapter for each layer
            adapted_prod_weight = torch.eye(self.width)
            for l in range(l1, l2):
                # compute the lora adapter for the lth layer
                Ql = V @ generate_diag_matrix(self.width, min(self.rank*(l-l1), self.width), min(self.rank*(l-l1+1), self.width)) @ V.T
                lora_adapter[l] = torch.inverse(frozen_prod_weight_l2L[l]) @ discrepancy_weight @ Ql @ torch.inverse(adapted_prod_weight)
                adapted_prod_weight = (adapted_m.linearlist[l].weight.data + lora_adapter[l]) @ adapted_prod_weight
                
                # update the lora weights in the adapter model
                U_Q, S_Q, V_Q = torch.svd(Ql)
                adapted_m.loralist[l].lora_A.data = U_Q @ torch.diag(S_Q)
                adapted_m.loralist[l].lora_B.data = V_Q
                
            # update the bias in the adapter model
            if self.use_bias:
                calibrate_bias = torch.zeros(self.width)
                adapted_prod_weight_rev = torch.eye(self.width)
                adapted_prod_weight_l2L = {(l2-1): adapted_prod_weight_rev}
                for l in range(l1+1, l2)[::-1]:
                    adapted_prod_weight_rev = adapted_prod_weight_rev @ adapted_m.linearlist[l].weight.data
                    adapted_prod_weight_l2L[l-1] = adapted_prod_weight_rev
                
                for l in range(l1, l2):
                    
                    # update the intermediate output
                    z = adapted_m.linearlist[l].weight.data @ z
                    
                    # update the bias in the adapter model
                    if l < l2-1:
                        # ensure all the relus are activated
                        adapted_m.linearlist[l].bias.data = - 2 * min(torch.min(z).item(), 0) * torch.ones(self.width)
                        
                        # update the intermediate output
                        z = z + adapted_m.linearlist[l].bias.data
                        
                        # update the calibrated bias
                        calibrate_bias = calibrate_bias + adapted_prod_weight_l2L[l] @ adapted_m.linearlist[l].bias.data
                    else:
                        # matching the bias in the target model
                        adapted_m.linearlist[l].bias.data = self.target_m.linearlist[i].bias.data - calibrate_bias
                        
        return adapted_m
                        
                    
    def eval(
        self,
        adapted_model,
        n_test,
    ):
        # generate random input from some Gaussian distribution
        x_test = torch.randn(n_test, self.width) 
        y_test = self.target_m(x_test).detach()
        y_test.requires_grad = False
            
        y_pred = adapted_model(x_test)
        loss = self.criterion(y_pred, y_test)
        
        return loss.item()