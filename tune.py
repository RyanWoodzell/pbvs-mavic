import os
import json
import optuna
import torch
import subprocess
from logger import logger

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.json')
TRAIN_SCRIPT = os.path.join(ROOT_DIR, 'src', 'train.py')

def load_base_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def run_trial(trial):
    """
    Optuna Trial function.
    Proposes hyperparameters, updates config.json, and runs training.
    """
    # 1. Propose Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
    kd_weight = trial.suggest_float("kd_weight", 0.1, 1.0)
    con_weight = trial.suggest_float("con_weight", 0.05, 0.5)
    alpha_mixup = trial.suggest_float("alpha_mixup", 0.0, 1.0)
    
    # We keep batch_size fixed or conditionally chosen to prevent OOMs
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    logger.info(f"\n🚀 Starting Trial {trial.number} 🚀")
    logger.info(f"Proposed Params: LR={lr:.5f}, WD={wd:.5f}, Batch={batch_size}, KD={kd_weight:.2f}")

    # 2. Update and Write config.json
    cfg = load_base_config()
    cfg['hyperparameters']['lr'] = lr
    cfg['hyperparameters']['wd'] = wd
    cfg['hyperparameters']['batch_size'] = batch_size
    cfg['hyperparameters']['kd_weight'] = kd_weight
    cfg['hyperparameters']['con_weight'] = con_weight
    cfg['hyperparameters']['alpha_mixup'] = alpha_mixup
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
        
    # 3. Execute Training via Subprocess (to safely isolate CUDA environments)
    try:
        # We run the DDP process exactly as a standard run
        result = subprocess.run(
            ['python', 'app.py'], 
            cwd=ROOT_DIR,
            capture_output=True,
            text=True
        )
        
        # 4. Parse Output to find the mathematically highest validation score achieved during this trial
        best_score_this_trial = 0.0
        
        # We scan standard out/error for the validation prints from train.py:
        # e.g., "📈 [VAL] Acc: 0.2800, AUROC: 0.7700, Score: 0.4025"
        for line in result.stdout.split('\n') + result.stderr.split('\n'):
            if "Score: " in line and "📈 [VAL]" in line:
                try:
                    score_str = line.split("Score: ")[1].strip()
                    score_val = float(score_str)
                    if score_val > best_score_this_trial:
                        best_score_this_trial = score_val
                except Exception:
                    pass
        
        logger.info(f"🏁 Trial {trial.number} Finished. Best Score: {best_score_this_trial:.4f}")
        
        # If the training crashed entirely (e.g. OOM), brutally punish the trial
        if best_score_this_trial == 0.0 and result.returncode != 0:
            logger.error(f"❌ Trial crashed. Return code: {result.returncode}")
            return 0.0
            
        return best_score_this_trial

    except Exception as e:
        logger.error(f"❌ Trial {trial.number} Failed Exceptionally: {str(e)}")
        return 0.0

if __name__ == "__main__":
    logger.info("🤖 Starting MAVIC-C Optuna Hyperparameter Turing Engine...")
    
    # We want to MAXIMIZE the Score metric
    study = optuna.create_study(
        study_name="PBVS_MAVIC_V2_Optimization",
        direction="maximize",
        storage="sqlite:///mavic_optuna.db",
        load_if_exists=True
    )
    
    # Run 50 trials (or User can Kill (Ctrl+C) anytime)
    try:
        study.optimize(run_trial, n_trials=50, gc_after_trial=True)
    except KeyboardInterrupt:
        logger.info("🛑 Optuna Search Manually Interrupted!")
        
    logger.info("🏆 OPTIMIZATION COMPLETE 🏆")
    best_trial = study.best_trial
    logger.info(f"Best Score: {best_trial.value:.4f}")
    logger.info(f"Best Hyperparameters: {best_trial.params}")
    
    # Auto-Restore the golden settings back to config.json
    logger.info("Injecting ultimate parameters back into config.json...")
    cfg = load_base_config()
    for k, v in best_trial.params.items():
        if k in cfg['hyperparameters']:
            cfg['hyperparameters'][k] = v
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    logger.info("✅ config.json permanently updated with Best Parameters.")
