from load_model import *
from copy import deepcopy
from tqdm import tqdm
from torch import optim
import argparse

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
        method = 'ours',
        batch_size = 5000,
        n_epochs = 1000,
        lr = 1e-3,
        n_test = 10000,
        weight_decay = 0,
        log_wandb = 0,
        init_mode = 'default',
    ):
        set_seed()
        
        self.width = width
        self.target_depth = target_depth
        self.frozen_depth = frozen_depth
        self.rank = rank
        self.use_bias = use_bias
        self.wandb = log_wandb
        
        self.init_models(std, activation, init_mode)
        
        self.criterion = nn.MSELoss()
        
        # perform finetune
        if method == 'ours':
            adapted_m = self.adapt_ours(batch_size = batch_size)
        elif method == 'sgd':
            adapted_m = self.adapt_sgd(
                n_epochs = n_epochs,
                batch_size = batch_size,
                lr = lr,
                weight_decay = weight_decay,
            )
        else:
            raise NotImplementedError(f"We only support ours and sgd for parameter method, and {method} is not supported.")
        
        # evaluate the adapted model
        self.eval(adapted_m, n_test = n_test)
        
    def init_models(
        self,
        std,
        activation,
        init_mode,
    ):
        # randomly initialize the target model
        self.target_m = FNN(
            depth = self.target_depth,
            width = self.width,
            rank = self.rank,
            std = std,
            use_bias = self.use_bias,
            apply_lora = False,
            activation = activation,
        )
        self.target_m.eval()
        
        # randomly initialize the frozen model
        self.frozen_m = FNN(
            depth = self.frozen_depth,
            width = self.width,
            rank = self.rank,
            std = std,
            use_bias = self.use_bias,
            apply_lora = True,
            activation = activation,
        )
        self.frozen_m.eval()
        
        if init_mode == 'uniform_singular_values':
            tdl = self.frozen_depth // self.target_depth
            for i in range(self.target_depth):
                # use range(l1, l2) layers in the adapted model to approximate the ith layer in the target model 
                l1 = i * tdl
                l2 = (i + 1) * tdl if i < self.target_depth - 1 else self.frozen_depth
                
                # compute the product of the frozen matrices
                frozen_prod_weight = torch.eye(self.width)
                for l in range(l1, l2):
                    frozen_prod_weight = self.frozen_m.linearlist[l].weight.data @ frozen_prod_weight 
                    
                # set the target weight
                discrepency_matrix = self.target_m.linearlist[i].weight.data - frozen_prod_weight
                self.target_m.linearlist[i].weight.data = frozen_prod_weight + torch.eye(self.width) * torch.mean(torch.svd(discrepency_matrix)[1])

    def adapt_sgd(
        self,
        n_epochs,
        batch_size,
        lr,
        weight_decay = 0,
    ):
        set_seed()
        
        adapted_m = deepcopy(self.frozen_m)
        adapted_m.train()
        
        # specify the lora adapter as the parameters to be optimized
        params = []
        for l in range(self.frozen_depth):
            params.append({'params': adapted_m.loralist[l].lora_A, 'lr': lr, 'weight_decay': weight_decay})
            params.append({'params': adapted_m.loralist[l].lora_B, 'lr': lr, 'weight_decay': weight_decay})
            
            if self.use_bias:
                params.append({'params': adapted_m.linearlist[l].bias, 'lr': lr, 'weight_decay': weight_decay})
            
        opt = optim.Adam(params)
            
        # Initialize tqdm
        iter_obj = tqdm(range(n_epochs))
        # finetuning
        for i in iter_obj:
            # generate random input from some Gaussian distribution
            x_train = torch.randn(batch_size, self.width) 
            y_train = self.target_m(x_train).detach()
            y_train.requires_grad = False
            
            y_pred = adapted_m(x_train)
            self.train_loss = self.criterion(y_pred, y_train)
            
            if self.wandb:
                wandb.log({'train_loss': self.train_loss.item()})
            
            opt.zero_grad()
            self.train_loss.backward()
            opt.step()
            
            # update tqdm description with current loss
            iter_obj.set_description(f"Loss of SGD: {self.train_loss.item():.4f}")      
            
        # validation
        adapted_m.eval()
        x_val =  torch.randn(batch_size, self.width) 
        y_val = self.target_m(x_val).detach()
        y_val.requires_grad = False
        
        y_pred = adapted_m(x_val)
        self.val_loss = self.criterion(y_pred, y_val)
        
        if self.wandb:
            wandb.log({'val_loss': self.val_loss.item()})
        else:
            print(f"Validation loss: {self.val_loss.item():.4f}")
        
        return adapted_m
    
    def adapt_ours(
        self,
        batch_size,
    ):
        set_seed()
        
        tdl = self.frozen_depth // self.target_depth
        adapted_m = deepcopy(self.frozen_m)
        adapted_m.train()
        
        if self.use_bias:
            # generate random input from some Gaussian distribution
            z = torch.randn(batch_size, self.width) 
        
        for i in range(self.target_depth):
            # use range(l1, l2) layers in the adapted model to approximate the ith layer in the target model 
            l1 = i * tdl
            l2 = (i + 1) * tdl if i < self.target_depth - 1 else self.frozen_depth
            
            # get the target and frozen weights for l1 to l2 layers
            target_weight = self.target_m.linearlist[i].weight.data
            frozen_weights = []
            for l in range(l1, l2):
                frozen_weights.append(adapted_m.linearlist[l].weight.data)
                
            lora_A, lora_B = our_construction(target_weight, frozen_weights, self.rank, self.wandb)
                
            for l in range(l1, l2):
                # update the lora weights in the adapter model
                adapted_m.loralist[l].lora_A.data = lora_A[l-l1]
                adapted_m.loralist[l].lora_B.data = lora_B[l-l1]
            
            # update the bias in the adapter model
            if self.use_bias:
                calibrate_bias = torch.zeros(self.width)
                adapted_prod_weight_rev = torch.eye(self.width)
                adapted_prod_weight_l2L = {(l2-1): adapted_prod_weight_rev}
                for l in range(l1+1, l2)[::-1]:
                    adapted_prod_weight_rev = adapted_prod_weight_rev @ (adapted_m.linearlist[l].weight.data + adapted_m.loralist[l].lora_A @ adapted_m.loralist[l].lora_B.T)
                    adapted_prod_weight_l2L[l-1] = adapted_prod_weight_rev
                
                for l in range(l1, l2):
                    
                    # update the intermediate output without the bias
                    z = z @ adapted_m.linearlist[l].weight.data.T + adapted_m.loralist[l].forward(z)
                    
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
        set_seed()
        
        adapted_model.eval()
        
        # generate random input from some Gaussian distribution
        x_test = torch.randn(n_test, self.width) 
        y_test = self.target_m(x_test).detach()
        y_test.requires_grad = False
            
        y_pred = adapted_model(x_test)
        loss = self.criterion(y_pred, y_test)
        
        self.test_loss = loss.item()
        
        if self.wandb:
            wandb.log({'test_loss': self.test_loss})
        else:
            print(f"Test loss: {self.test_loss:.4f}")
            
            
class approx_tfn:
    def __init__(
        self,
        embed_dim,
        n_head,
        depth,
        rank,
        batch_size,
        seq_length,
        method, 
        n_epochs,
        lr,
        weight_decay,
        log_wandb = 0,
        std = .25,
        n_test = 10000,
    ):
        set_seed()
        
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.rank = rank
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.wandb = log_wandb
        
        self.init_models(std)
        
        self.criterion = nn.MSELoss()
        
        # perform finetune
        if method == 'ours':
            adapted_m = self.adapt_ours()
        elif method == 'sgd':
            adapted_m = self.adapt_sgd(
                n_epochs,
                batch_size,
                lr,
                weight_decay,
            )
        else:
            raise NotImplementedError(f"We only support ours and sgd for parameter method, and {method} is not supported.")
        
        # evaluate the adapted model
        self.eval(adapted_m, n_test = n_test)
            
        
    def init_models(
        self,
        std,
    ):
        # randomly initialize the target model
        self.target_m = TFN(
            embed_dim = self.embed_dim,
            n_head = self.n_head,
            depth = self.depth,
            rank = self.rank,
            std = std,
            apply_lora = False,
        )
        self.target_m.eval()
        
        self.frozen_m = TFN(
            embed_dim = self.embed_dim,
            n_head = self.n_head,
            depth = self.depth,
            rank = self.rank,
            std = std,
            apply_lora = True,
        )
        self.frozen_m.eval()
        
    def adapt_sgd(
        self,
        n_epochs,
        batch_size,
        lr, 
        weight_decay = 0,
    ):
        set_seed()
        
        adapted_m = deepcopy(self.frozen_m)
        adapted_m.train()
        
        # specify the lora adapter as the parameters to be optimized
        params = []
        for l in range(self.depth):
            # lora on the attention layers
            attention_lora = adapted_m.tfblist[l].attention.loralist
            for i in range(len(attention_lora)):
                params.append({'params': attention_lora[i].lora_A, 'lr': lr, 'weight_decay': weight_decay})
                params.append({'params': attention_lora[i].lora_B, 'lr': lr, 'weight_decay': weight_decay})
                
            # lora on the feedforward layers
            if l == self.depth - 1:
                W2_lora = adapted_m.tfblist[l].W2_lora
                params.append({'params': W2_lora.lora_A, 'lr': lr, 'weight_decay': weight_decay})
                params.append({'params': W2_lora.lora_B, 'lr': lr, 'weight_decay': weight_decay})
            
        # lora on the output layer 
        output_lora = adapted_m.output_layer_lora
        params.append({'params': output_lora.lora_A, 'lr': lr, 'weight_decay': weight_decay})
        params.append({'params': output_lora.lora_B, 'lr': lr, 'weight_decay': weight_decay})
        
        opt = optim.Adam(params)
        
        # Initialize tqdm
        iter_obj = tqdm(range(n_epochs))
        # finetuning
        for i in iter_obj:
            # generate random input from some Gaussian distribution
            X_train = torch.randn(batch_size, self.embed_dim, self.seq_length)
            Y_train = self.target_m(X_train).detach()
            Y_train.requires_grad = False
            
            Y_pred = adapted_m(X_train)
            self.train_loss = self.criterion(Y_pred, Y_train)
            
            if self.wandb:
                wandb.log({'train_loss': self.train_loss.item()})
                
            opt.zero_grad()
            self.train_loss.backward()
            opt.step()
            
            # update tqdm description with current loss
            iter_obj.set_description(f"Loss of SGD: {self.train_loss.item():.4f}")
            
        # validation
        adapted_m.eval()
        X_val = torch.randn(batch_size, self.embed_dim, self.seq_length)
        Y_val = self.target_m(X_val).detach()
        Y_val.requires_grad = False
        
        Y_pred = adapted_m(X_val)
        self.val_loss = self.criterion(Y_pred, Y_val)
        
        if self.wandb:
            wandb.log({'val_loss': self.val_loss.item()})
        else:
            print(f"Validation loss: {self.val_loss.item():.4f}")
            
        return adapted_m
    
    def adapt_ours(
        self,
    ):
        set_seed()
        
        adapted_m = deepcopy(self.frozen_m)
        adapted_m.train()
        
        iter_obj = tqdm(range(self.depth))
        
        ######################### DEBUG ONLY #########################  
        # x = torch.randn(self.batch_size, self.embed_dim, self.seq_length)
        # target_x, adapted_x = x, x
        ################################################################
        
        # iterating through each transformer block
        for l in iter_obj:
            adapted_attention = adapted_m.tfblist[l].attention
            target_attention = self.target_m.tfblist[l].attention

            # iterating through each head
            for h in range(self.n_head):
                # consider component: W_K^h.T * W_Q^h
                # get the target matrix and the frozen matrix
                # treat  W_Q^h as the first layer, and W_K^h.T as the second layer
                frozen_kq = [adapted_attention.Wq[h].data, adapted_attention.Wk[h].data.T] 
                if l == 0:
                    target_kq = target_attention.Wk[h].data.T @ target_attention.Wq[h].data
                
                else:
                    calibrate_ff2 = self.target_m.tfblist[l-1].feed_forward[1].weight.data @ torch.inverse(adapted_m.tfblist[l-1].feed_forward[1].weight.data)
                    target_kq = calibrate_ff2.T @ target_attention.Wk[h].data.T @ target_attention.Wq[h].data @ calibrate_ff2
                    
                lora_A, lora_B = our_construction(target_kq, frozen_kq, self.rank, self.wandb)
            
                # update the adapted for W_Q^h
                adapted_attention.loralist[h*4].lora_A.data = lora_A[0]
                adapted_attention.loralist[h*4].lora_B.data = lora_B[0]
                
                # update the adapted for W_K^h
                adapted_attention.loralist[h*4+1].lora_A.data = lora_B[1]
                adapted_attention.loralist[h*4+1].lora_B.data = lora_A[1]
                
                # consider component: W_O^h * W_V^h
                # get the target matrix and the frozen matrix
                # treat  W_V^h as the first layer, and W_O^h as the second layer
                frozen_ov = [adapted_attention.Wv[h].data, adapted_attention.Wo[h].data]
                calibrate_ff1 = torch.inverse(adapted_m.tfblist[l].feed_forward[0].weight.data) @ self.target_m.tfblist[l].feed_forward[0].weight.data
                if l == 0:
                    target_ov = calibrate_ff1 @ target_attention.Wo[h].data @ target_attention.Wv[h].data
                else:
                    target_ov = calibrate_ff1 @ target_attention.Wo[h].data @ target_attention.Wv[h].data @ calibrate_ff2
                
                lora_A, lora_B = our_construction(target_ov, frozen_ov, self.rank, self.wandb)
                
                # update the adapted for W_V^h
                adapted_attention.loralist[h*4+2].lora_A.data = lora_A[0]
                adapted_attention.loralist[h*4+2].lora_B.data = lora_B[0]
                
                # update the adapted for W_O^h
                adapted_attention.loralist[h*4+3].lora_A.data = lora_A[1]
                adapted_attention.loralist[h*4+3].lora_B.data = lora_B[1]
                
            # update the bias in the feedforward network
            # match the bias of the first feedforward layer
            adapted_m.tfblist[l].feed_forward[0].bias.data = self.target_m.tfblist[l].feed_forward[0].bias.data
            # match the bias of the second feedforward layer
            if l < self.depth - 1:
                adapted_m.tfblist[l].feed_forward[1].bias.data = adapted_m.tfblist[l].feed_forward[1].weight.data @ torch.inverse(self.target_m.tfblist[l].feed_forward[1].weight.data) @ self.target_m.tfblist[l].feed_forward[1].bias.data      

            else:
                # match the output layer
                # consider the component: W_o W_{2L}
                target_ol = self.target_m.output_layer.weight.data @ self.target_m.tfblist[l].feed_forward[1].weight.data
                frozen_ol = [adapted_m.tfblist[-1].feed_forward[1].weight.data, adapted_m.output_layer.weight.data]
                
                lora_A, lora_B = our_construction(target_ol, frozen_ol, self.rank, self.wandb)
                
                # update the adapted W_2l
                adapted_m.tfblist[l].W2_lora.lora_A.data = lora_A[0]
                adapted_m.tfblist[l].W2_lora.lora_B.data = lora_B[0]
                
                # update the adapted W_o
                adapted_m.output_layer_lora.lora_A.data = lora_A[1]
                adapted_m.output_layer_lora.lora_B.data = lora_B[1]
                
                # update the bias 
                updated_output_layer_weight = adapted_m.output_layer.weight.data + adapted_m.output_layer_lora.lora_A.data @ adapted_m.output_layer_lora.lora_B.data.T
                adapted_m.tfblist[l].feed_forward[1].bias.data = torch.inverse(updated_output_layer_weight) @ self.target_m.output_layer.weight.data @ self.target_m.tfblist[l].feed_forward[1].bias.data
                
            ######################### DEBUG ONLY ######################### 
            # compute the difference between the intermediate ouput H_l of the target model and the adapted model
            # target_h = self.target_m.tfblist[l].forward_ff1(target_x)
            # adapted_h = adapted_m.tfblist[l].forward_ff1(adapted_x)
            # self.train_loss = torch.mean(torch.norm(target_h - adapted_h, dim = 1)).item()
            # iter_obj.set_description(f"Loss of Ours: {self.train_loss:.4f}")
            # print(self.train_loss)
            
            # target_x = self.target_m.tfblist[l].forward_ff2(target_x)
            # adapted_x = adapted_m.tfblist[l].forward_ff2(adapted_x)
            # if l < self.depth - 1:
            #     print(torch.mean(torch.norm(target_x - self.target_m.tfblist[l].feed_forward[1].weight.data @ torch.inverse(adapted_m.tfblist[l].feed_forward[1].weight.data) @ adapted_x, dim = 1)).item())
            # else:
            #     target_x = self.target_m.output_layer.weight.data @ target_x
            #     adapted_x = adapted_m.output_layer.weight.data @ adapted_x
            #     print(torch.mean(torch.norm(target_x - adapted_x, dim = 1)).item())
            ################################################################
        
        return adapted_m
    
    def eval(
        self,
        adapted_model, 
        n_test,
    ):
        # generated random input from some Gaussian distribution
        X_test = torch.randn(n_test, self.embed_dim, self.seq_length)
        Y_test = self.target_m(X_test).detach()
        Y_test.requires_grad = False
        
        Y_pred = adapted_model(X_test)
        loss = self.criterion(Y_pred, Y_test)
        
        self.test_loss = loss.item()
        
        if self.wandb:
            wandb.log({'test_loss': self.test_loss})
        else:
            print(f"Test loss: {self.test_loss:.4f}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--target_depth', type=int, default=1)
    parser.add_argument('--frozen_depth', type=int, default=2)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--use_bias', type=int, default=1, choices = [0,1])
    parser.add_argument('--activation', type=str, default='relu', choices = ['relu', 'linear'])
    parser.add_argument('--std', type=float, default=.25)
    parser.add_argument('--method', type=str, default='ours', choices = ['ours', 'sgd'])
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--init_mode', type=str, default='default', choices = ['default', 'uniform_singular_values'])
    
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--seq_length', type=int, default=10)

    parser.add_argument('--exp', type=str, default='fnn', choices = ['fnn', 'tfn'])
    parser.add_argument('--wandb', type=int, default=0, choices = [0,1])
    
    args = parser.parse_args()
    
    # print experiment configuration
    args_dict = vars(args)
    print('Experiment Setting:')
    for key, value in args_dict.items():
        print(f"| {key}: {value}")
    
    # initialize wandb
    if args.wandb:
        wandb.init(
            project = "lora-theory",
            group =  args.exp,
            entity = 'lee-lab-uw-madison',
            job_type = args.init_mode,
            config = args,
        )
    
    # run the experiment
    if args.exp == 'fnn':
        approx_fnn(
            width = args.width,
            target_depth = args.target_depth,
            frozen_depth = args.frozen_depth,
            rank = args.rank,
            use_bias = args.use_bias,
            activation = args.activation,
            std = args.std,
            method = args.method,
            batch_size = args.batch_size,
            n_epochs = args.n_epochs,
            lr = args.lr,
            n_test = args.n_test,
            weight_decay = args.weight_decay,
            log_wandb = args.wandb,
            init_mode = args.init_mode,
        )

    elif args.exp == 'tfn':
        if (args.target_depth != args.frozen_depth) and (args.method == 'ours'):
            raise NotImplementedError(f"our method only support the case where the target depth is equal to the frozen depth for TFN. Please try sgd instead.")
        if args.init_mode == 'uniform_singular_values':
            raise NotImplementedError(f"uniform_singular_values is not supported for TFN. Please try default instead.")
            
        approx_tfn(
            embed_dim = args.width,
            n_head = args.n_head,
            depth = args.target_depth,
            rank = args.rank,
            batch_size = args.batch_size,
            seq_length = args.seq_length,
            method = args.method,
            n_epochs = args.n_epochs,
            lr = args.lr,
            weight_decay = args.weight_decay,
            log_wandb = args.wandb,
            std = args.std,
            n_test = args.n_test,
        )

    if args.wandb:
        wandb.finish()
