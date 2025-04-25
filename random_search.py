import subprocess
import itertools
import os
import random
import datetime
import argparse

# Modelos disponibles y sus hiperparámetros a considerar
model_hyperparams = {
    "HyperSaR": {
        "num_layer": [1, 2, 3, 4],
        "edge_dropout": [0.05 ,0.1, 0.15, 0.2, 0.25, 0.3],
        "loss_weight": [0.001, 0.005, 0.01, 0.05, 0.1]
    },
    "LightGCN": {
        "num_layer": [1, 2, 3, 4],
        "edge_dropout": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    },
    "JSR": {
        "num_layer": [1, 2, 3, 4],
        "weight_dropout": [0.0 , 0.1, 0.15, 0.2, 0.25, 0.3],
        "loss_weight": [0.001, 0.005, 0.01, 0.05, 0.1]
    },
    "DeepFM": {
        "num_layer": [1, 2, 3, 4],
        "weight_dropout": [0.0 , 0.1, 0.15, 0.2, 0.25, 0.3]
    }
}

# Argumentos CLI
parser = argparse.ArgumentParser(description="Random Search para modelos de recomendación")
parser.add_argument("--model", type=str, choices=model_hyperparams.keys(), required=True,
                    help="Nombre del modelo a ejecutar")
parser.add_argument("--dataset", type=str, default="lastfm", help="Nombre del dataset a usar")
parser.add_argument("--num_trials", type=int, default=5, help="Cantidad de combinaciones aleatorias a probar")
parser.add_argument("--num_epoch", type=int, default=100, help="Número de épocas por entrenamiento")
args = parser.parse_args()

selected_model = args.model
dataset = args.dataset
num_trials = args.num_trials
num_epoch = args.num_epoch

# Crear carpeta para logs si no existe
log_dir = f"random_search_logs/{selected_model}"
os.makedirs(log_dir, exist_ok=True)

# Extraer hiperparámetros
hyperparam_space = model_hyperparams[selected_model]
param_keys = list(hyperparam_space.keys())

# Generar combinaciones aleatorias
all_combinations = list(itertools.product(*[hyperparam_space[k] for k in param_keys]))
random.shuffle(all_combinations)

# Ejecutar cada combinación
for i, values in enumerate(all_combinations[:num_trials]):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"{selected_model}_trial{i+1}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_name)

    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--model", selected_model,
        "--num_epoch", str(num_epoch)
    ]

    for key, value in zip(param_keys, values):
        cmd.extend([f"--{key}", str(value)])

    print(f"[INFO] Ejecutando trial {i+1}/{num_trials}")
    print("Comando:", ' '.join(cmd))
    
    # Marcar inicio total
    total_start = datetime.datetime.now()
    train_start = total_start

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        eval_start = None  # Se marca cuando empieza la evaluación
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)

            # Marcar cuándo comienza la evaluación
            if "Starting evaluation..." in line and eval_start is None:
                eval_start = datetime.datetime.now()

        process.wait()

    # Marcar fin de todo
    total_end = datetime.datetime.now()

    # Calcular duraciones
    if eval_start:
        train_duration = eval_start - train_start
        eval_duration = total_end - eval_start
    else:
        train_duration = total_end - train_start
        eval_duration = datetime.timedelta(0)

    total_duration = total_end - total_start

    print(f"[INFO] Trial {i+1} completado")
    print(" - 🎯 Hiperparámetros seleccionados:")
    for key, value in zip(param_keys, values):
        print(f"   - {key}: {value}")
    print(f" - 🏋️ Entrenamiento: {train_duration}")
    print(f" - 📊 Evaluación: {eval_duration}")
    print(f" - ⏱️ Tiempo total: {total_duration}")
    print(f" - 📁 Log: {log_path}\n")
