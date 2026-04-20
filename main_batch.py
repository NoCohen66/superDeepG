
import argparse
import pandas as pd
import torch
from utils import load_yaml_file, check_config_keys
from experiment.config.wrapper import set_WrapperNN
import math
from get_bounds import get_bounds, extract_results_df
from utils import lirpa_format_transformation
import time
import sys
import layer.lip.grad_utils as grad_utils
from experiment.normalize_utils import NormalizeInput

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def main():
    
    """
    Main function:
    1) Parse a YAML config file.
    2) Load dataset (a), model (b) and solver settings (c).
    3) Loop over images and apply the solver's method.
    4) Save the results.
    """
    # -------------------------------------------------------------------------
    # 1) Parse a YAML config file.
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Superdeepg")
    parser.add_argument("--config_path", type=str,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--num_images", type=int, default=2,
                        help="Number of images to load from CSV.")
    args = parser.parse_args()
    
    # Check if config path was specified
    if args.config_path == None:
        print("Usage: python main_batch.py --config_path <path_to_yaml> ")
        sys.exit(1)
    
    config = load_yaml_file(args.config_path)
    check_config_keys(config, ["dataset", "model", "solver"])

    # -------------------------------------------------------------------------
    # 2)a. Load dataset 
    # -------------------------------------------------------------------------
    dataset_cfg = config["dataset"]
    dataset_path = dataset_cfg["path"]                 
    process_module = dataset_cfg["path_process"]      
    process_func_name = dataset_cfg["process_dataset_func"]  
    num_images = dataset_cfg["num_images"]  
    dataset_module = __import__(process_module, fromlist=[process_func_name])
    process_dataset_func = getattr(dataset_module, process_func_name)
    images, labels, n_classes, means, stddevs, img_shape = process_dataset_func(dataset_path,
                                                                                max_images=num_images)
    print(f"[INFO] Loaded {len(images)} images from {dataset_path}.")
    print(f"[INFO] Number of classes assumed: {n_classes}.")
    print(f"[INFO] Number of images: {num_images}.")
    

    # -------------------------------------------------------------------------
    # 2)b. Load model 
    # -------------------------------------------------------------------------

    model_cfg = config["model"]
    weights_path = model_cfg["weights_path"] 
    archi = model_cfg["architectures"][0]
    archi_path = archi["path"]         
    weight_loader_name = archi["weight_loader"]  
    # Dynamically import the function that loads weights
    model_module = __import__(archi_path, fromlist=[weight_loader_name])
    weight_loader_func = getattr(model_module, weight_loader_name)
    net = weight_loader_func(weights_path, device).to(device)
    net.eval()
    print(f"[INFO] Model loaded from: {weights_path}")


    # -------------------------------------------------------------------------
    # 2)c. Load solver settings
    # -------------------------------------------------------------------------
    solver_cfg = config["solver"]
    theta_batch_size = solver_cfg["theta_batch_size"]
    methods = solver_cfg["methods"]              # e.g., ["CROWN"]
    if len(methods) > 1:
        raise NotImplementedError("Support for multiple methods is not yet available.")
    alpha_config = None
    KKT_samples = solver_cfg["KKT_samples"]      # e.g., 100
    KKT_samples_batch = solver_cfg["KKT_samples_batch"]      # e.g., 100
    split_second_step = solver_cfg["split_second_step"] # e.g., 1000
    split_second_step_batch = solver_cfg["split_second_step_batch"] # e.g., 1000
 
    if "display_info" in solver_cfg.keys():
        display_info = solver_cfg["display_info"] #  False or True
    else:
        display_info = False
    if "alpha_tr" in solver_cfg.keys():
        alpha_tr = solver_cfg["alpha_tr"]
    else: 
        alpha_tr = False
    if "use_C" in solver_cfg.keys():
        use_C = solver_cfg["use_C"]
    else: 
        use_C = True
    
    if "spec_im" in solver_cfg.keys(): #specify some images to run
        spec_im = solver_cfg["spec_im"]
    else: 
        spec_im = [i for i in range(len(labels))]

    
    transformation, theta_min, theta_max, eps_init = solver_cfg["config"]
    if transformation == "rotate":
        theta_min = theta_min * math.pi / 180.0
        theta_max = theta_max * math.pi / 180.0
        eps_init = eps_init * math.pi / 180.0

    
    if transformation != "translate":
        thetas, eps, inside_split = lirpa_format_transformation(
            theta_min, theta_max, eps_init, display=True
        )
        thetas_batch = torch.tensor(thetas).unsqueeze(1)# (B = interval_length, 1)
    
    else:
        thetas_center = []
        thetas_center_card = []
        for i in range(len(theta_min)):
            thetas, eps, inside_split = lirpa_format_transformation(
            theta_min[i], theta_max[i], eps_init, display=True )
            thetas_center.append(torch.tensor(thetas).unsqueeze(1))
            thetas_center_card.append(torch.tensor(thetas))
        thetas_batch = torch.cartesian_prod(*thetas_center_card)
      
   
    start_global_time = time.time()

    # Lipschitz 
    if transformation != "translate":
        interval_size = eps
    else:
        interval_size = eps
    if type(theta_min) == int or type(theta_max) == float : 
        LB, UB, transformation_list = [theta_min], [theta_max], [transformation]
    else:
        LB, UB, transformation_list = [theta_min[0]], [theta_max[0]], [transformation]
    zipped = sorted(zip(transformation_list, LB, UB))
    trans, lbs, ubs = zip(*zipped)
    lb = torch.tensor(lbs, dtype=torch.float64).unsqueeze(0).to(device)
    ub = torch.tensor(ubs, dtype=torch.float64).unsqueeze(0).to(device)
    
    if ("mnist" in dataset_path) or ("MNIST" in dataset_path):
        H, W = (28, 28)
        C = 1
    elif ("cifar" in dataset_path) or ("CIFAR" in dataset_path):
        H, W = (32, 32)
        C = 3

    elif ("tinyimagenet-1-255-ccibp.pt" in weights_path):
        H, W = (64,64)
        C = 3
    else:
        H, W = (56,56)
        C = 3
    
    tr_grad = grad_utils.compute_grad_tr_im(
        C=C, H=H, W=W, transformations=list(trans),
        lb=lb, ub=ub, interval_size=interval_size, device=device,
    ) # [1, 2, 1, cand, C, H, W]
    
 
    
    print(f"[INFO]: there are {inside_split} inside_split")
 

    # -------------------------------------------------------------------------
    # 4) Loop over images and apply the solver's method.
    # -------------------------------------------------------------------------
 
    results = []
    for idx, (image, label) in enumerate(zip(images, labels)):
        if idx in spec_im: #expected image_shape (1, C, H, W)
        
            start = time.time()
            interp_grad = grad_utils.compute_grad_interp(image.squeeze(0).permute(1,2,0), device=device)
            L = grad_utils.compute_grad_product_transformation_interpolation(tr_grad, interp_grad) # 1, C, H, W
            
            print(f"\n[INFO] Processing image #{idx} with label={label} for theta in [{theta_min}, {theta_max}]")
            image = image.to(device) #(B, C, W, H)
            KKT_tensor = torch.Tensor([KKT_samples]) # From solver config
            KKT_tensor_batch = torch.Tensor([KKT_samples_batch]) # From solver config
            split_second_step = torch.Tensor([split_second_step])
            split_second_step_batch = torch.Tensor([split_second_step_batch])
            theta_eps_tensor = torch.Tensor([eps])
            lip_constant = L # This is only the lipschitz constant of the transformation
            normalize = NormalizeInput(mean=means, std=stddevs, channels=img_shape[0])
            with torch.no_grad():
                pred_class = torch.argmax(net(normalize(image)), dim=1).item()
                robust_without_pert = (pred_class == label)

            if not robust_without_pert:
                print("----------------")
                print(f"\n[INFO] Image #{idx} is not robust without perturbation.")
                print("----------------")
                robust = pd.NA
                incorrect = True
                df = extract_results_df(idx, robust, label, incorrect, "",  torch.tensor([0])[None], eps, 0, lb=torch.tensor([[0]*n_classes]), ub=torch.tensor([[0]*n_classes]), n_classes=n_classes)
                results.append(df)
            else:
                dfs_for_image = []

                model_rot_inter = set_WrapperNN(
                        model=net,
                        image=image,
                        KKT_samples=KKT_tensor,
                        KKT_samples_batch=KKT_tensor_batch,
                        L=lip_constant,
                        N=split_second_step,
                        N_batch=split_second_step_batch,
                        transformation=transformation,
                        theta_eps=theta_eps_tensor, 
                        method=methods[0], 
                        alpha_tr=alpha_tr, 
                        means=means, 
                        stddevs=stddevs, 
                        img_shape=img_shape
                        )
                model_rot_inter.eval()
                for start_idx in range(0, thetas_batch.size(0), theta_batch_size):

                    thetas_selected = thetas_batch[start_idx:start_idx+theta_batch_size, :] #batch,d
                    
                    df, robust = get_bounds(
                        model=model_rot_inter,
                        theta_input=thetas_selected, #batch difference
                        eps=eps,
                        methods=methods,
                        true_label=label,
                        n_classes=n_classes,
                        image_index=idx,
                        display=display_info,
                        alpha_crown_config=alpha_config, 
                        transformation=transformation, 
                        use_C=use_C
                    )
                
                    dfs_for_image.append(df)
                   
                    if robust is not True:
                          print(f"[INFO] Image #{idx} found non-robust for a batch of thetas.")
                          break
                  
                end = time.time()
                exec_time_tot = end - start
                if len(dfs_for_image) > 0:
                    df_img = pd.concat(dfs_for_image)
                    results.append(df_img)
                                
    # -------------------------------------------------------------------------
    # 5) Save the results.
    # -------------------------------------------------------------------------
    if len(results) > 0:
        all_df = pd.concat(results)
        file_name = args.config_path.split(".")[0].split("/")[-1]
        output_path = f"experiment/results/{file_name}_results.csv"
        all_df.to_csv(output_path, index=False)
        print(f"[INFO] Results saved to: {output_path}")
    else:
        print("No results to save.")

    end_global_time = time.time()
    print(f"The complete time (!!!with csv results files writing!!!) for {num_images} is {round(end_global_time-start_global_time, 2)} seconds.")

if __name__ == "__main__":
    main()