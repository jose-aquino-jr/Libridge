"""
Sistema de Reconhecimento de Libras - Menu Principal
"""
import os
import sys
import subprocess

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def mostrar_menu():
    limpar_tela()
    print("=" * 50)
    print("SISTEMA DE RECONHECIMENTO DE LIBRAS")
    print("=" * 50)
    print("\n1. Coletar dados para treinamento (com sequencias)")
    print("2. Treinar modelo de IA")
    print("3. Deteccao em tempo real")
    print("4. Estatisticas do dataset")
    print("5. Sair")
    print("=" * 50)

def executar_script(script_nome):
    try:
        subprocess.run([sys.executable, script_nome])
    except Exception as e:
        print(f"Erro ao executar {script_nome}: {e}")
        input("\nPressione Enter para continuar...")

def mostrar_estatisticas():
    import pandas as pd
    import os
    
    if os.path.exists('dados/dataset_libras.csv'):
        df = pd.read_csv('dados/dataset_libras.csv')
        print(f"\nEstatisticas do Dataset:")
        print(f"Total de amostras (sequencias): {len(df)}")
        print(f"Numero de letras: {df['letra'].nunique()}")
        print(f"Frames por sequencia: {int((len(df.columns)-1)/63)}")  # 21 landmarks * 3 coordenadas
        print("\nAmostras por letra:")
        for letra in sorted(df['letra'].unique()):
            count = len(df[df['letra'] == letra])
            print(f"  Letra {letra}: {count} sequencias")
    else:
        print("\nDataset nao encontrado!")
    
    input("\nPressione Enter para continuar...")

def main():
    while True:
        mostrar_menu()
        opcao = input("\nEscolha uma opcao: ").strip()
        
        if opcao == '1':
            executar_script('coletor_dados.py')
        elif opcao == '2':
            executar_script('treinador.py')
        elif opcao == '3':
            executar_script('detector_tempo_real.py')
        elif opcao == '4':
            mostrar_estatisticas()
        elif opcao == '5':
            print("\nAte logo!")
            break
        else:
            print("\nOpcao invalida!")
            input("Pressione Enter para continuar...")

if __name__ == "__main__":
    main()