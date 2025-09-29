import os

def buscar_archivos_wav(directorio):
    archivos_wav = []

    # Caminar por el directorio
    for root, dirs, files in os.walk(directorio):
        for file in files:
            # Comprueba si el archivo termina en .wav
            if file.endswith('.wav'):
                # Unir la ruta del directorio con el nombre del archivo
                path_absoluto = os.path.join(root, file)
                
                # Comprueba si existe un archivo correspondiente con extensión .pho.npy
                path_phonpy = path_absoluto.replace('.wav', '.pho.npy')
                if os.path.exists(path_phonpy):
                    archivos_wav.append(path_absoluto)
    
    return archivos_wav

def guardar_en_txt(archivos_wav, nombre_txt='archivos_wav.txt'):
    # Abrir el archivo txt en modo de escritura
    with open(nombre_txt, 'w') as f:
        for path in archivos_wav:
            path_phonpy = path.replace('.wav', '.pho.npy')
            # Escribir cada path en una línea nueva
            f.write(path + "|" + path_phonpy + "\n")

directorio = '/mnt/aholab/inigop/corpus/neutral/kiko_eu/'  # Inserta tu directorio aquí
archivos_wav = buscar_archivos_wav(directorio)
guardar_en_txt(archivos_wav)
