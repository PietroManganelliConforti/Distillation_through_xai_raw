#!/usr/bin/env python3

import os
import subprocess

# Directory contenente gli elementi
DIR = "work/project/save/imagenette/"

# Itera su ogni elemento della directory
for element in os.listdir(DIR):
    element_path = os.path.join(DIR, element)
    state_dict_path = os.path.join(element_path, "state_dict.pth")
    
    # Verifica che sia una directory e che il file state_dict.pth esista
    if os.path.isdir(element_path) and os.path.isfile(state_dict_path):
        print(f"Eseguendo comando per: {element}")
        try:
            subprocess.run([
                "python3", "work/project/cam_for_dist.py", "--m_pth", f"save/imagenette/{element}/state_dict.pth"
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"Errore con {element}, passando al prossimo.")