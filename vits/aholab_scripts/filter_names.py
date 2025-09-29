import argparse

def filtrar_lineas(archivo_entrada, archivo_salida, sub1, sub2):
    with open(archivo_entrada, 'r', encoding='utf-8') as archivo:
        lineas = archivo.readlines()

    lineas_filtradas = [linea for linea in lineas if sub1 in linea or sub2 in linea]

    with open(archivo_salida, 'w', encoding='utf-8') as archivo:
        archivo.writelines(lineas_filtradas)

def main():
    parser = argparse.ArgumentParser(description='Filtrar líneas que contienen las subpalabras "kiko" y "aintz"')
    parser.add_argument('ini', help='Nombre del archivo de entrada')
    parser.add_argument('out', help='Nombre del archivo de salida')
    args = parser.parse_args()

    sub1="kiko"
    sub2="aintz"
    filtrar_lineas(args.ini, args.out, sub1, sub2)

if __name__ == '__main__':
    main()
