import os
import subprocess

caminho = r"C:\Users\xunin\OneDrive\Documentos\IA\projeto_libras"

os.chdir(caminho)

subprocess.run(["python", "main.py"])

input("\nPressione Enter para sair...")