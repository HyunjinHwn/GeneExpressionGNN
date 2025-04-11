import torch
import copy
import scipy.stats as stats
import time
import random 
import warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')

def spearman_correlation_approx(x: torch.Tensor, y: torch.Tensor):
    """Approximate correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    def _get_ranks(x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp).to(x.device)
        ranks[tmp] = torch.arange(len(x)).to(x.device)
        return ranks
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)
   
class OverallError(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.se = torch.nn.MSELoss(reduction='none')
        self.eps = eps
        
    def forward(self,yhat,y):
        se = self.se(yhat,y)
        loss = torch.sqrt(se.sum(dim=1) + self.eps)
        return loss.mean()
    
def train(model, data, config_name, graph_name, loss_func, epochs, lr, wd):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, verbose=False)
    valid_step = 10
    save_prediction = f'prediction/{graph_name}/{config_name}.pkl'
    patience = 0
    print_loss = 0.
    best_loss, best_corr = 1e10, -1.
    start_time = time.time()
    for e in range(1, epochs+1):
        model.train()
        # to boost the training speed, we (re)sample 500 genes from the training set for each epoch
        sampled_idx = random.sample(data.idx_train, 500)
        for j in sampled_idx:
            data.features[:,-1] = data.l1000_features[:,j]
            output = model(data.features)
            loss = loss_func(output, data.Y[j,:].unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()
            print_loss += loss.item()
        
        if e==1 or e % valid_step == 0: # validation
            result_dict_val = evaluate(model, data, data.idx_val, loss_func, approx=True)
            if result_dict_val['corr'] > best_corr:
                best_epoch = e
                best_loss = result_dict_val['loss']
                best_corr =  result_dict_val['corr']
                best_model = copy.deepcopy(model)
                # if best, update test set performance
                result_dict_test = evaluate(best_model, data, data.idx_test, loss_func, approx=True)
                print(f"Epoch {e:3d} Test Loss\t{result_dict_test['loss']:.3f}\tCorr\t{result_dict_test['corr']:.4f}" )
                # save temporal best 1) checkpoint and 2) inferred gene expressions for test set
                torch.save(best_model.state_dict(), f'checkpoints/{graph_name}/{config_name}.pt')
                with open(save_prediction, 'wb') as f:
                    pickle.dump(np.array(result_dict_test['prediction']).squeeze().T, f)
                patience = 0
            else:
                patience += 1
            end_time = time.time()
            print_loss = print_loss/len(sampled_idx)/valid_step
            print(f'Epoch {e:3d} Train Loss: {print_loss:.4f},  lr:{optimizer.param_groups[0]["lr"]:.6f}\t'
                                +f'Valid Loss: {best_loss:.4f},  corr: {best_corr:.4f}\t'
                                +f'Time {end_time - start_time:.2f}sec')
            
            print_loss = 0
            scheduler.step(result_dict_val['loss'])
        if patience*valid_step > 200: break
    print('Validation: ', best_epoch, best_loss, best_corr)
    print(f"Validation best at {best_epoch} epoch, Loss\t{best_loss:.3f}\tCorr\t{best_corr}" )
    
    # Test set performance
    result_dict_test = evaluate(best_model, data, data.idx_test, loss_func, approx=False)
    torch.save(np.array(result_dict_test['prediction']).squeeze().T, save_prediction)
    print(f"Test Corr\t{result_dict_test['corr']:.4f}\tOverallError\t{result_dict_test['rmse_loss']:.3f}\tUsedLoss\t{result_dict_test['loss']:.3f}" )
    print(f"Test Corr\t{result_dict_test['corr_std']:.4f}\tOverallError\t{result_dict_test['rmse_loss_std']:.3f}" )
    return best_model
    
             
def evaluate(model, data, idx, loss_func, approx=False):
    model.eval()
    error_func = OverallError()
    print_loss = 0
    spearman_corr_list = torch.zeros(len(idx)).cuda()
    pred = []
    rmse_loss_list = []
    for i, j in enumerate(idx):
        data.features[:,-1] = data.l1000_features[:,j]
        output = model(data.features)
        pred.append(output.detach().cpu().tolist())
        loss = loss_func(output, data.Y[j,:].unsqueeze(0))
        print_loss += loss.item()
        rmse_loss_list.append(error_func(output, data.Y[j,:].unsqueeze(0)).item())
        if approx:
            result = spearman_correlation_approx(data.Y[j,:].detach(), output.detach().squeeze())
        else:
            result = stats.spearmanr(data.Y[j,:].unsqueeze(-1).detach().cpu().numpy(), 
                                     output.detach().cpu().numpy().T).statistic
            
        spearman_corr_list[i] = result
    print_loss /= len(idx)
    result_dict = dict()
    result_dict['loss'] = print_loss
    result_dict['rmse_loss'] = np.mean(rmse_loss_list)
    result_dict['rmse_loss_std'] = np.std(rmse_loss_list)
    result_dict['corr'] = torch.mean(spearman_corr_list)
    result_dict['corr_std'] = torch.std(spearman_corr_list)
    result_dict['prediction'] = pred
    return result_dict
