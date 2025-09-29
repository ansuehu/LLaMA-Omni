import os
import argparse
import soundfile as sf
import numpy as np

def convert_int16_to_float32(input_file):
    # Leer archivo .wav
    data, samplerate = sf.read(input_file, dtype='int16')
    
    # Convertir int16 a float32
    float_data = data.astype(np.float32) / 32768.0  # Dividir por 32768 para normalizar al rango [-1, 1]
    
    # Escribir archivo .wav en formato float-32 (reemplazando el original)
    sf.write(input_file, float_data, samplerate, subtype='FLOAT')

def process_directory(directory):
    # Listar todos los archivos en el directorio
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            full_path = os.path.join(directory, filename)
            print(f"Procesando {full_path}...")
            convert_int16_to_float32(full_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir todos los archivos .wav de int-16 a float-32 en un directorio dado.")
    parser.add_argument('directory', type=str, help="Directorio que contiene los archivos .wav a convertir.")
    
    args = parser.parse_args()
    
    process_directory(args.directory)
