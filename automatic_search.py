import os
import subprocess
import itertools
import numpy as np
import re
import json
from datetime import datetime
import optuna
import json
import numpy as np
from datetime import datetime

def save_results(results, path, filename="hyperparameter_search_results.json"):
    """Save all results to a JSON file"""
    save_path = os.path.join(path, filename)
    assert os.path.exists(path), f"Path {path} does not exist. Please create it first."
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")

def run_experiment(lr, loss_cam_weight, variance_weight, epochs = 30, layer = "model.layer4", 
                   xai_shape=2, dataset_name="caltech256", model_name="resnet18"):
    """Run a single experiment with given hyperparameters"""


    command = [
        "python3", "work/project/train.py",
        "--dataset", str(dataset_name),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", "32",
        "--model", str(model_name),
        "--pretrained",
        "--xai_poisoning",
        "--xai_shape", str(xai_shape),
        "--loss_cam_weight", str(loss_cam_weight),
        "--info_text", "_TEST_",
        "--layer", str(layer),
        "--continue_option",
        "--variance_weight", str(variance_weight),
        "--variance_fixed_weight", "0.0",
        #"--load_weights_pretrained_path", "work/project/save/caltech256/resnet18_/0.005_10_pretrained/state_dict.pth"
    ]

    if dataset_name == "caltech256" and model_name == "resnet18":
        command.append("--load_weights_pretrained_path")
        command.append("work/project/save_past_26_6/caltech256/resnet18_/0.005_10_pretrained/state_dict.pth")
    
    print(f"Running experiment: lr={lr}, loss_cam_weight={loss_cam_weight}, variance_weight={variance_weight}")
    result = subprocess.run(command, capture_output=True, text=True)

    # Get the path from the last line of stdout
    if result.stdout.strip():
        mypath = result.stdout.strip().splitlines()[-1]
        
        # Remove "PATH: " prefix if present
        if mypath.startswith("PATH: "):
            mypath = mypath[6:]  # Remove "PATH: " (6 characters)
        
        print("Il mio path è:", mypath)

    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None, None, None
    
    if not os.path.exists(mypath):
        print(f"Warning: Path {mypath} does not exist.")
        return None, None, None
    
    results_file = os.path.join(mypath, "results.txt")

    if not os.path.exists(results_file):
        print(f"Warning: Results file {results_file} does not exist. Check the training script output.")
        return None, None, None
    
    try:
        with open(results_file, 'r') as f:
            results_content = f.read()
            results_lines = results_content.strip().splitlines()
            print(f"Results lines: {results_lines}")
            
            # Extract metrics from the lines
            top1_accuracy = None
            mse_mcam_t = None
            
            for line in results_lines:
                if line.startswith("Test accuracy:"):
                    top1_accuracy = float(line.split(":")[1].strip())
                elif line.startswith("MSE_mCAM_t:"):
                    mse_mcam_t = float(line.split(":")[1].strip())
            
            print(f"Extracted top1_accuracy: {top1_accuracy}, MSE_mCAM_t: {mse_mcam_t}")
            
            if top1_accuracy is None or mse_mcam_t is None:
                print("Warning: Could not extract metrics from results file.")
                return None, None, mypath
                
            return top1_accuracy, mse_mcam_t, mypath
            
    except Exception as e:
        print(f"Error reading results file: {e}")
        return None, None, mypath



def find_results_file(base_path="work/project/save"):
    """Find the most recent results file"""
    # Look for result files in the save directory
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.txt') and ('result' in file.lower() or 'log' in file.lower()):
                return os.path.join(root, file)
    return None

def extract_metrics_from_file(file_path):
    """Extract top1_accuracy and MSE_mCAM_t from results file"""
    if not file_path or not os.path.exists(file_path):
        return None, None
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        top1_accuracy = None
        mse_mcam_t = None
        
        # Look for exact patterns from your results format
        top1_match = re.search(r'top1_accuracy:\s*([0-9]+\.?[0-9]*)', content)
        if top1_match:
            top1_accuracy = float(top1_match.group(1))
            
        mse_match = re.search(r'MSE_mCAM_t:\s*([0-9]+\.?[0-9]*)', content)
        if mse_match:
            mse_mcam_t = float(mse_match.group(1))
                
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None
        
    return top1_accuracy, mse_mcam_t

def is_better_result(new_top1, new_mse, best_top1, best_mse, threshold=60.0):
    """
    Determine if new result is better than current best.
    Primary criterion: top1_accuracy must be above threshold (60%)
    Secondary criterion: lowest MSE_mCAM_t among those above threshold
    """
    # If no previous best result
    if best_top1 is None:
        return new_top1 is not None and new_top1 >= threshold
    
    if new_top1 is None:
        return False
    
    # Current best is above threshold
    if best_top1 >= threshold:
        # New result must also be above threshold to be considered
        if new_top1 >= threshold:
            # Both above threshold: choose the one with lower MSE
            if best_mse is None:
                return new_mse is not None
            if new_mse is None:
                return False
            return new_mse < best_mse
        else:
            # New result below threshold, keep current best
            return False
    else:
        # Current best is below threshold
        if new_top1 >= threshold:
            # New result above threshold is always better
            return True
        else:
            # Both below threshold: choose higher accuracy, then lower MSE
            if new_top1 > best_top1:
                return True
            elif new_top1 < best_top1:
                return False
            else:
                # Same accuracy: choose lower MSE
                if best_mse is None:
                    return new_mse is not None
                if new_mse is None:
                    return False
                return new_mse < best_mse

def main():
    # Define hyperparameter search space
    lr_values = [5e-5]
    loss_cam_weight_values = [5.0]
    variance_weight_values = [0.15]
    epochs = 50
    layer = "model.layer4"
    xai_shape = 2
    model_name = "resnet18"
    # Target threshold for top1_accuracy
    accuracy_threshold = 60.0
    
    # Generate all combinations
    param_combinations = list(itertools.product(lr_values, loss_cam_weight_values, variance_weight_values))
    total_experiments = len(param_combinations)
    
    print(f"Starting hyperparameter search with {total_experiments} combinations...")
    print(f"TARGET: Find top1_accuracy >= {accuracy_threshold}% with lowest MSE_mCAM_t")
    print(f"Search space:")
    print(f"  Learning rates: {lr_values}")
    print(f"  Loss CAM weights: {loss_cam_weight_values}")
    print(f"  Variance weights: {variance_weight_values}")
    print("-" * 60)
    
    results = []
    best_top1_accuracy = None
    best_mse_mcam_t = None
    best_params = None
    candidates_above_threshold = []
    path = "work/project/save/"
    dataset_name = "caltech256"
    
    for i, (lr, loss_cam_weight, variance_weight) in enumerate(param_combinations):
        print(f"\nExperiment {i+1}/{total_experiments}")
        
        # Run the experiment
        top1_accuracy, mse_mcam_t, path = run_experiment(lr, loss_cam_weight, 
                                                         variance_weight , epochs, 
                                                         layer, xai_shape, dataset_name, model_name)


        # Store results
        experiment_result = {
            'experiment_id': i + 1,
            'lr': lr,
            'loss_cam_weight': loss_cam_weight,
            'variance_weight': variance_weight,
            'top1_accuracy': top1_accuracy,
            'mse_mcam_t': mse_mcam_t,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(experiment_result)
        
        if top1_accuracy is not None:
            print(f"Top1 Accuracy: {top1_accuracy:.4f}%")
            if mse_mcam_t is not None:
                print(f"MSE_mCAM_t: {mse_mcam_t:.6f}")
            
            # Check if this result meets the threshold
            if top1_accuracy >= accuracy_threshold:
                candidates_above_threshold.append(experiment_result.copy())
                print(f"✓ CANDIDATE FOUND! Above {accuracy_threshold}% threshold")
            
            # Update best result using the comparison function
            if is_better_result(top1_accuracy, mse_mcam_t, best_top1_accuracy, best_mse_mcam_t, accuracy_threshold):
                best_top1_accuracy = top1_accuracy
                best_mse_mcam_t = mse_mcam_t
                best_params = {
                    'lr': lr,
                    'loss_cam_weight': loss_cam_weight,
                    'variance_weight': variance_weight
                }
                if best_top1_accuracy >= accuracy_threshold:
                    print(f"*** NEW BEST CANDIDATE! Top1 Accuracy: {best_top1_accuracy}%, MSE_mCAM_t: {best_mse_mcam_t} ***")
                else:
                    print(f"*** NEW BEST (below threshold): Top1 Accuracy: {best_top1_accuracy:.4f}%, MSE_mCAM_t: {best_mse_mcam_t} ***")
        else:
            print("Warning: Could not extract metrics from results")
        
        # Save intermediate results
        save_results(results,path.split("//")[0], filename="hyperparameter_search_results.json")
        
        # Show current status
        num_candidates = len(candidates_above_threshold)
        print(f"Candidates above {accuracy_threshold}%: {num_candidates}")
        if best_params:
            status = "✓ TARGET MET" if best_top1_accuracy and best_top1_accuracy >= accuracy_threshold else "Below threshold"
            print(f"Current best [{status}]: lr={best_params['lr']}, ",
                  f"loss_cam_weight={best_params['loss_cam_weight']}, ",
                  f"variance_weight={best_params['variance_weight']}, ",
                  f"top1_accuracy={best_top1_accuracy:.4f}%, ",
                  f"mse_mcam_t={best_mse_mcam_t:.6f}")
    
    # Final results
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETED")
    print("="*80)
    
    print(f"Total candidates above {accuracy_threshold}%: {len(candidates_above_threshold)}")
    
    if best_params:
        if best_top1_accuracy and best_top1_accuracy >= accuracy_threshold:
            print(f"\n  TARGET ACHIEVED!")
            print(f"Best hyperparameters (top1_accuracy >= {accuracy_threshold}% with lowest MSE_mCAM_t):")
        else:
            print(f"\n   No results above {accuracy_threshold}% found.")
            print(f"Best hyperparameters found (highest accuracy available):")
            
        print(f"  Learning rate: {best_params['lr']}")
        print(f"  Loss CAM weight: {best_params['loss_cam_weight']}")
        print(f"  Variance weight: {best_params['variance_weight']}")
        print(f"  Top1 accuracy: {best_top1_accuracy:.4f}%")
        if best_mse_mcam_t is not None:
            print(f"  MSE_mCAM_t: {best_mse_mcam_t:.6f}")
        

    else:
        print("No valid results found. Please check the training script and output format.")
    
    # Sort and display candidates above threshold
    if candidates_above_threshold:
        candidates_above_threshold.sort(key=lambda x: (x['mse_mcam_t'] if x['mse_mcam_t'] is not None else float('inf')))
        print(f"\nAll candidates above {accuracy_threshold}% (sorted by MSE_mCAM_t):")
        for i, result in enumerate(candidates_above_threshold):
            print(f"{i+1}. lr={result['lr']}, loss_cam_weight={result['loss_cam_weight']}, "
                  f"variance_weight={result['variance_weight']}, "
                  f"top1_accuracy={result['top1_accuracy']:.4f}%, "
                  f"mse_mcam_t={result['mse_mcam_t']:.6f if result['mse_mcam_t']}")
    
    # Sort all results for general reference
    valid_results = [r for r in results if r['top1_accuracy'] is not None]
    valid_results.sort(key=lambda x: (-x['top1_accuracy'], x['mse_mcam_t'] if x['mse_mcam_t'] is not None else float('inf')))
    
    print(f"\nTop 5 overall results:")
    for i, result in enumerate(valid_results[:5]):
        status = "✓" if result['top1_accuracy'] >= accuracy_threshold else "✗"
        print(f"{i+1}. {status} lr={result['lr']}, loss_cam_weight={result['loss_cam_weight']}, "
              f"variance_weight={result['variance_weight']}, "
              f"top1_accuracy={result['top1_accuracy']:.4f}%, "
              f"mse_mcam_t={result['mse_mcam_t']:.6f if result['mse_mcam_t']}")
    
    # Save final results
    final_results = {
        'search_completed': datetime.now().isoformat(),
        'total_experiments': total_experiments,
        'accuracy_threshold': accuracy_threshold,
        'candidates_above_threshold': len(candidates_above_threshold),
        'target_achieved': best_top1_accuracy is not None and best_top1_accuracy >= accuracy_threshold,
        'best_params': best_params,
        'best_top1_accuracy': best_top1_accuracy,
        'best_mse_mcam_t': best_mse_mcam_t,
        'all_candidates_above_threshold': candidates_above_threshold,
        'all_results': results
    }
    
    save_results(final_results, path, "final_hyperparameter_search_results.json")
    print(f"\nAll results saved to: final_hyperparameter_search_results.json")








import numpy as np
import optuna
import json
from datetime import datetime

def save_trial_results(study, save_path):
    """Salva i risultati correnti dello studio in formato JSON"""
    results = []
    
    for t in study.trials:
        trial_data = {
            "trial_number": t.number,
            "params": t.params,
            "values": t.values,
            "accuracy": t.user_attrs.get("acc"),
            "mse": t.user_attrs.get("mse"),
            "state": t.state.name
        }
        results.append(trial_data)
    
    # Trova i trial Pareto ottimali
    pareto_trials = []
    if hasattr(study, 'best_trials') and study.best_trials:
        for t in study.best_trials:
            if t.user_attrs.get("acc") is not None and t.user_attrs.get("mse") is not None:
                pareto_trials.append({
                    "trial_number": t.number,
                    "params": t.params,
                    "values": t.values,
                    "accuracy": t.user_attrs["acc"],
                    "mse": t.user_attrs["mse"],
                    "state": t.state.name
                })
    
    summary = {
        "datetime": datetime.now().isoformat(),
        "total_trials": len(results),
        "completed_trials": len([t for t in results if t["state"] == "COMPLETE"]),
        "pareto_trials_count": len(pareto_trials),
        "pareto_trials": pareto_trials,
        "all_trials": results,
        "objectives": ["minimize MSE", "maximize Accuracy"]
    }
    
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

def objective(trial):
    # Parametri di ottimizzazione
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    loss_cam_weight = trial.suggest_float("loss_cam_weight", 0.03, 4.0)
    variance_weight = trial.suggest_categorical("variance_weight", [0.0, 0.15, 0.3])

    epochs = 10
    layer = "model.layer4"
    layer_name = "layer4"
    xai_shape = 0
    dataset_name = "caltech256"
    model_name = "resnet50"

    # Esegui l'esperimento
    acc, mse, _ = run_experiment(lr, loss_cam_weight, variance_weight,
                                  epochs, layer, xai_shape, dataset_name, 
                                    model_name)
    
    if acc is None or mse is None:
        return float("inf"), float("inf")

    # Salva i valori negli attributi utente per riferimento
    trial.set_user_attr("acc", acc)
    trial.set_user_attr("mse", mse)
    
    # Salva i risultati dopo ogni trial
    save_path = "work/project/save/" + dataset_name + "/" + model_name + "/" + layer_name + "/" +str(xai_shape)
    save_path = os.path.join(save_path, "optuna_results.json")
    save_trial_results(trial.study, save_path)
    
    print(f"Trial {trial.number}: Acc={acc:.3f}%, MSE={mse:.6f} - Results saved")

    # Restituisce (mse, -acc) per minimizzare MSE e massimizzare accuracy
    return mse, -acc

def main_optuna(n_trials=30):
    print(f"Running Optuna multi-objective optimization with {n_trials} trials")
    print("Objectives: Minimize MSE & Maximize Accuracy")

    # Crea lo studio multi-obiettivo
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # minimize MSE, minimize -accuracy
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    
    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Mostra i migliori trial (Pareto front)
    print(f"\nPareto optimal trials found: {len(study.best_trials)}")
    
    for i, trial in enumerate(study.best_trials):
        acc = trial.user_attrs.get("acc")
        mse = trial.user_attrs.get("mse")
        print(f"Trial {i+1}: Accuracy={acc:.3f}%, MSE={mse:.6f}")
        print(f"  Params: {trial.params}")
        print()
    
    return study

if __name__ == "__main__":
    study = main_optuna(n_trials=100)
    #main()