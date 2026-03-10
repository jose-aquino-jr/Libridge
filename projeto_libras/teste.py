import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Versão do MediaPipe:", mp.__version__)
print("\nVerificando módulos disponíveis:")

# Lista todos os atributos do mediapipe
print("\nAtributos do módulo mediapipe:")
for attr in dir(mp):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Tenta importar o solutions pelo caminho correto
try:
    from mediapipe.python.solutions import hands
    print("\n✅ Sucesso! from mediapipe.python.solutions import hands")
except Exception as e:
    print(f"\n❌ Erro: {e}")

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    print("\n✅ Sucesso! mp.solutions.hands")
except Exception as e:
    print(f"\n❌ Erro: {e}")