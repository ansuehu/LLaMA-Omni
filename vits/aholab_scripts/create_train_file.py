import os
import random
import sys
import soundfile

def buscar_archivos_no_vacios(directorio, extension):
    archivos_no_vacios = []

    # Caminar por el directorio
    for root, dirs, files in os.walk(directorio):
        for file in files:
            # Comprueba si el archivo tiene la extensión deseada
            if file.endswith(extension):
                # Unir la ruta del directorio con el nombre del archivo
                path_absoluto = os.path.join(root, file)

                # Comprueba si el archivo no está vacío (peso mayor a 0KB)
                if os.path.getsize(path_absoluto) > 0:
                    archivos_no_vacios.append(path_absoluto)

    return archivos_no_vacios

def buscar_archivos_wav_en_carpeta(directorio, identificador, bit_type, specs):
    archivos_wav = []

    for archivo_wav in buscar_archivos_no_vacios(directorio, '.wav'):
        # Construir el nombre del archivo .pho.npy correspondiente
        archivo_pho_npy = archivo_wav.replace('.wav', '.pho.npy')
        archivo_spec = os.path.join(specs, os.path.basename(archivo_wav)).replace('.wav', '.spec.pt')
        archivos_wav.append((archivo_wav, archivo_pho_npy, identificador, bit_type, archivo_spec))

    return archivos_wav

def buscar_archivos_wav_en_carpetas_eu(directorio, directorio_spec):
    archivos_wav = []
    carpetas_eu = []

    # Caminar por el directorio
    for root, dirs, files in os.walk(directorio):
        for dir in dirs:
            # Comprueba si el nombre de la carpeta termina en '_eu'
            if dir.endswith('_eu'):
                # Obtiene la ruta completa de la carpeta
                carpeta_eu = os.path.join(root, dir)
                carpetas_eu.append(carpeta_eu)

    # Genera identificadores únicos para cada carpeta _eu
    identificadores = {carpeta: i for i, carpeta in enumerate(carpetas_eu)}

    for carpeta_eu in carpetas_eu:
        identificador = identificadores[carpeta_eu]
        bit_type = get_random_wav_bit(carpeta_eu)
        specs = os.path.join(directorio_spec, os.path.basename(carpeta_eu))
        # Busca archivos WAV en esta carpeta
        archivos_wav.extend(buscar_archivos_wav_en_carpeta(carpeta_eu, identificador, bit_type, specs))

    return archivos_wav

def buscar_archivos_wav_en_carpetas_finetuning(carpeta, directorio_spec, id):
    archivos_wav = []

    bit_type = get_random_wav_bit(carpeta)
    specs = os.path.join(directorio_spec, os.path.basename(carpeta))
    if not os.path.exists(specs):
        os.makedirs(specs, exist_ok=True)
    # Busca archivos WAV en esta carpeta
    archivos_wav.extend(buscar_archivos_wav_en_carpeta(carpeta, id, bit_type, specs))

    return archivos_wav

def get_random_wav_bit(folder_path):
    """Devuelve un archivo .wav aleatorio de la carpeta especificada."""
    
    # Listar todos los archivos en el directorio
    all_files = os.listdir(folder_path)
    
    # Filtrar solo los archivos .wav
    wav_files = [f for f in all_files if f.endswith('.wav')]
    
    # Si no hay archivos .wav en el directorio, devolver None
    if not wav_files:
        print("No se encontraron archivos .wav en el directorio especificado.")
        return None
    
    # Elegir aleatoriamente un archivo .wav y devolver su ruta completa
    aleatorio = random.choice(wav_files)
    return get_wav_bit_depth(os.path.join(folder_path, aleatorio))

def get_wav_bit_depth(filename):
    """Devuelve la profundidad de bits (bit depth) de un archivo .wav usando soundfile."""

    info = soundfile.info(filename)
    subtype = info.subtype
    # Extraer la profundidad de bits a partir del subtipo
    if '16' in subtype:
        return 16
    elif 'FLOAT' in subtype:
        return 32
    else: 
        print("Problema con el bit_depth: No reconocido")
        return -1

def guardar_en_txt(archivos_wav, nombre_txt='archivos_wav.txt'):
    # Obtener la ruta del directorio actual del script
    directorio_actual = os.path.dirname(__file__)
    ruta_txt = os.path.join(directorio_actual, nombre_txt)
    
    # Mezclar las rutas de los archivos de manera aleatoria
    random.shuffle(archivos_wav)
    
    # Abrir el archivo txt en modo de escritura
    with open(ruta_txt, 'w') as f:
        for archivo_wav, archivo_pho_npy, identificador, bit_type, specs in archivos_wav:
            if bit_type==16:
                num = 32768.0
            elif bit_type==32:
                num = 1
            else:
                print("Problemas con el bit type, el archivo {archivo_wav} no es ni float-32 ni int-16")
            # Escribir cada path en una línea nueva con el identificador
            f.write(f"{archivo_wav}|{identificador}|{archivo_pho_npy}|{num}|{specs}\n")





directorio = '/mnt/aholab/inigop/corpus/personalized/163/recording_vits'  # Inserta tu directorio aquí
directorio_spec = '/mnt/aholab/inigop/corpus/personalized/163/recording_vits' # Inserta donde quieres que se guarden los espectrogramas (DEBES DE TENER PERMISOS DE ESCRTURA!!)
# archivos_wav = buscar_archivos_wav_en_carpetas_eu(directorio, directorio_spec)
archivos_wav = buscar_archivos_wav_en_carpetas_finetuning(directorio, directorio_spec, 11)
guardar_en_txt(archivos_wav)