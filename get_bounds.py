import torch 
from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
from auto_LiRPA.perturbations import PerturbationLpNorm
import time
from auto_LiRPA.operators.convolution import BoundConv
from layer.transformation_interpolation import BoundTransformationInterpolation
from collections import defaultdict
import pandas as pd
import logging
register_custom_op("custom::TransformationInterpolation", BoundTransformationInterpolation)

logging.getLogger('onnx').setLevel(logging.ERROR)
logging.getLogger('torch.onnx').setLevel(logging.ERROR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_bounds(model=None, theta_input=None, eps=None, methods=None, true_label='3', n_classes=10, image_index=None, regression=False, display=True, alpha_crown_config=None, transformation=None, use_C=None):
    start_time = time.time()
    theta_input = theta_input.float().to(device) #batch, d
    lirpa_model = BoundedModule(model, theta_input, bound_opts={"conv_mode": "matrix"})
    
    norm = float("inf")
    ptb = PerturbationLpNorm(norm = norm, eps = eps)
    input_lirpa = BoundedTensor(theta_input, ptb)

    # Check if incorrect without perturbation
    if isinstance(true_label, torch.Tensor):
        true_label = true_label.item()
    incorrect = False
    pred = lirpa_model(input_lirpa.to(device))
    label_pred = torch.argmax(pred, dim=1).cpu().detach().numpy()
    if all(label_pred_i != true_label for label_pred_i in label_pred):
        incorrect = True
    dfs = []

    if not incorrect: 
        for method in methods:  

            if alpha_crown_config is not None:
                lirpa_model.set_bound_opts(alpha_crown_config)
            if use_C:
                # When use_C is True, we build a spec matrix C to encode margins f_true − f_k for all k diff true_label
                # It computes lower upper bounds on each class margin via compute_bounds(bla bla, C=C)
                # In our case, the batch dim corresponds to different theta values, so C is repeated for each theta input
                other_classes = [k for k in range(n_classes) if k != true_label]   # classes diff true_label
                n_specs = len(other_classes) #  n_classes-1
                C = torch.zeros((theta_input.shape[0], n_specs, n_classes), device=theta_input.device)
                C[:, :, true_label] = 1.0                         # 1 sur la colonne true_label
                C[:, torch.arange(n_specs), torch.tensor(other_classes)] = -1.0  # −1 otherwise
                try:
                    lb, ub = lirpa_model.compute_bounds(x=(input_lirpa,), method=method.split()[0],C=C) # lb,ub shape [theta_input, n_classes - 1] 
                    # We want for each theta_input, that diff true_label - other_classes >0
                    robust_per_theta = (lb > 0).all(dim=1)  
                    end_time = time.time()
                    exec_time = end_time-start_time
                    exec_time_per_theta = exec_time/theta_input.size(0)
                    # Specific for 2D
                    if (theta_input.shape[1] == 2):
                        theta_input = theta_input[:,0]
              
                    for i in range(len(theta_input)):  #len(theta_input) = theta_batch
                        robust = robust_per_theta[i].item()
                        theta_input_i = theta_input[i]
                        lb_i = lb[i][None]
                        ub_i = ub[i][None]
                        df =  extract_results_df(image_index, robust, true_label, incorrect, method,  theta_input_i, eps, exec_time_per_theta, lb_i, ub_i, n_classes-1)
                        dfs.append(df)
                        if display: 
                            status = "✅ robust" if robust else "not robust"
                            print(f"Theta[{i}] = {theta_input_i.item():.4f}   —   {status}")
                            for spec_idx, k in enumerate(other_classes):
                                l = lb[i, spec_idx].item()
                                u = ub[i, spec_idx].item()
                                print(f"    margin f_{true_label}−f_{k}: {l:8.3f} <= ... <= {u:8.3f}")
                            print()  
                except ValueError as e:
                    print("ValueError", e)
                    for method in methods:
                        robust = pd.NA
                        df = extract_results_df(image_index, robust, true_label, incorrect, method,  theta_input, eps, exec_time=0, lb=torch.tensor([[0]*n_classes]), ub=torch.tensor([[0]*n_classes]), n_classes=n_classes)
                        dfs.append(df)

            else:
                lb, ub = lirpa_model.compute_bounds(x=(input_lirpa,), method=method.split()[0])#, return_A=True, needed_A_dict=required_A)
                labels_onehot = torch.nn.functional.one_hot(torch.tensor([true_label]), num_classes=n_classes).to(device) # (1, n_classes)
                interval_outputs = lb * labels_onehot + ub * torch.logical_not(labels_onehot) #(batch, n_classes)
                _, max_class_predicted = torch.max(interval_outputs, 1)
                robust_interval = (max_class_predicted == true_label)
                end_time = time.time()
                exec_time = end_time-start_time
                exec_time_per_theta = exec_time/theta_input.size(0)

                # Specific for 2D
                if (theta_input.shape[0] == 1) and (theta_input.shape[1] == 2):
                    theta_input = theta_input[:,0]
            
                for i in range(len(theta_input)):
                    robust = robust_interval[i].item()
                    theta_input_i = theta_input[i]
                    lb_i = lb[i][None]
                    ub_i = ub[i][None]
                    incorrect = 0
                    df =  extract_results_df(image_index, robust, true_label, incorrect, method,  theta_input_i, eps, exec_time_per_theta, lb_i, ub_i, n_classes)
                    dfs.append(df)
                    if display:
                        print(f"Computing image #{image_index}")
                        print(f"Theta in [{theta_input_i.item() - eps:10f}, {theta_input_i.item() + eps:10f}]")
                        print("Top-1 prediction {} ground-truth {} without perturbation".format(max_class_predicted, true_label))
                        if robust:
                            print("With perturbation now, it is robust. ")
                        else:
                            print("With perturbation now, it is NOT robust. ")
                        for j in range(n_classes):
                            indicator = '(ground-truth)' if j == true_label else ''
                            print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
                                j=j, l=lb_i[0][j].item(), u=ub_i[0][j].item(), ind=indicator))
            
                        print()

    else:
        print("incorrect")
        for method in methods:
            robust = pd.NA
            df = extract_results_df(image_index, robust, true_label, incorrect, method,  theta_input, eps, exec_time=0, lb=torch.tensor([[0]*n_classes]), ub=torch.tensor([[0]*n_classes]), n_classes=n_classes)
            dfs.append(df)
    return(pd.concat(dfs), robust)


def extract_results_df(image_index, robust, true_label, incorrect, method, theta_input, eps, exec_time, lb, ub, n_classes):
    """
    Extracts bounds into a pandas DataFrame for analysis.
    
    Args:
        theta_input (torch.Tensor): The input tensor to the model.
        eps (float): The perturbation epsilon value.
        lb (torch.Tensor): Lower bounds of the predictions.
        ub (torch.Tensor): Upper bounds of the predictions.
        n_classes (int): Number of classes in the classification task.

    Returns:
        pd.DataFrame: A DataFrame containing the bounds for each class along with input and epsilon.
    """
    try:
        lb = lb.cpu().detach().numpy()
        ub = ub.cpu().detach().numpy()
    except:
        lb = lb.cpu().numpy()
        ub = ub.cpu().numpy()

    data = []
    row = {
        "image_index":image_index,
        "robust":robust,
        "incorrect":incorrect,
        "true_label":true_label,
        "eps": eps, 
        "exec_time": exec_time, 
        "method":method
    }
    if theta_input.squeeze(0).shape == torch.Size([1]):
        row["theta_input"] = theta_input.item()
        row["theta_lb"] = theta_input.item() - eps
        row["theta_ub"] = theta_input.item() + eps

    
    for j in range(n_classes):
        row[f"class_{j}_lb"] = lb[0, j]
        row[f"class_{j}_ub"] = ub[0, j]
    
    data.append(row)
    df = pd.DataFrame(data)

    return df