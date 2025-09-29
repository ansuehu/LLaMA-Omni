import random
import sys

def shuffle_file_lines(filename):
    # Leer el contenido del archivo
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Desordenar las lineas
    random.shuffle(lines)

    # Escribir las lineas desordenadas de vuelta al archivo
    with open(filename, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python shuffle_lines.py <nombre_del_archivo>")
        sys.exit(1)

    file_name = sys.argv[1]
    shuffle_file_lines(file_name)
    print(f"Las lineas del archivo {file_name} han sido desordenadas!")
