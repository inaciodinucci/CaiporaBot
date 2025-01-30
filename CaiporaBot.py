import os
import subprocess
import urllib.request
import sys
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama
import logging  

MODEL_DIR = Path.home() / "modelos"
MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MODEL_URL = f"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/{MODEL_FILENAME}"

DEPENDENCIAS = ["llama-cpp-python", "transformers", "accelerate", "tqdm"]

CONTEXTO = """Você é o CaiporaBot, um assistente especializado em técnicas avançadas de sobrevivência na Amazônia.
Você sempre fala em Português do Brasil. 
Sua função é fornecer respostas claras, objetivas e altamente técnicas sobre os seguintes tópicos:
- Técnicas de sobrevivência: métodos comprovados para obtenção de fogo, purificação de água, construção de abrigos e obtenção de alimentos.
- Flora e fauna amazônicas: identificação de plantas comestíveis, medicinais e perigosas, além de comportamento de animais selvagens.
- Perigos da selva: prevenção de ataques de animais, identificação de riscos ambientais (como enchentes e tempestades) e primeiros socorros básicos.
Se você não tiver resposta para a pergunta não invente, diga sua funcionalidade.
Dê respostas diretas em português, com foco em técnicas práticas e científicas, sempre em no máximo 10 linhas. Priorize informações que possam salvar vidas em situações reais.
"""

logging.getLogger("llama_cpp").setLevel(logging.ERROR)

def instalar_dependencias():
    """Verifica e instala automaticamente as dependências"""
    print("🔧 Verificando dependências...")
    pacotes_faltantes = [pkg for pkg in DEPENDENCIAS if subprocess.run([sys.executable, "-m", "pip", "show", pkg], capture_output=True).returncode != 0]
    
    if pacotes_faltantes:
        print(f"📦 Instalando: {', '.join(pacotes_faltantes)}...")
        subprocess.run([sys.executable, "-m", "pip", "install", *pacotes_faltantes], check=True)
    else:
        print("✅ Todas as dependências estão instaladas.")

def progresso_download(url, destino):
    """Baixa o arquivo com barra de progresso"""
    response = urllib.request.urlopen(url)
    tamanho_total = int(response.headers.get("content-length", 0))
    
    with open(destino, "wb") as arquivo, tqdm(
        total=tamanho_total, unit="B", unit_scale=True, unit_divisor=1024, desc="📥 Baixando modelo"
    ) as barra:
        for dados in iter(lambda: response.read(8192), b""):
            arquivo.write(dados)
            barra.update(len(dados))

def baixar_modelo():
    """Baixa o modelo Mistral 7B Instruct se não existir ou estiver incompleto"""
    if MODEL_PATH.exists():
        print(f"⚠️ Modelo encontrado: {MODEL_PATH}")
        tamanho_local = MODEL_PATH.stat().st_size
        tamanho_remoto = int(urllib.request.urlopen(MODEL_URL).headers.get("content-length", 0))
        
        if tamanho_local == tamanho_remoto:
            print("✅ O modelo está completo.")
            return
        
        print("❌ Modelo incompleto! Baixando novamente...")
        MODEL_PATH.unlink()
    
    print("📥 Iniciando download do modelo...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    progresso_download(MODEL_URL, MODEL_PATH)
    
    if MODEL_PATH.stat().st_size != int(urllib.request.urlopen(MODEL_URL).headers.get("content-length", 0)):
        print("❌ Erro: O modelo foi baixado incorretamente. Tente rodar o script novamente.")
        sys.exit(1)
    
    print(f"✅ Download concluído: {MODEL_PATH}")

def carregar_modelo():
    """Carrega o modelo Mistral 7B sem logs excessivos"""
    print("🚀 Carregando modelo...")
    os.environ["LLAMA_LOG_LEVEL"] = "ERROR"
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=32768,
        n_threads=os.cpu_count(),
        n_gpu_layers=32,
        verbose=False
    )

def gerar_resposta(pergunta, modelo):
    prompt = f"{CONTEXTO}\nPergunta: {pergunta}\nResposta:"
    resposta = modelo(prompt, max_tokens=512, stop=["Pergunta:", "Resposta:"], temperature=0.2, top_p=0.9, repeat_penalty=1.2)["choices"][0]["text"]
    return resposta.strip()

def iniciar_chat():
    instalar_dependencias()
    baixar_modelo()
    modelo = carregar_modelo()
    
    ascii_art = """
   _____          _____ _____   ____  _____           
  / ____|   /\   |_   _|  __ \ / __ \|  __ \    /\    
 | |       /  \    | | | |__) | |  | | |__) |  /  \   
 | |      / /\ \   | | |  ___/| |  | |  _  /  / /\ \  
 | |____ / ____ \ _| |_| |    | |__| | | \ \ / ____ \ 
  \_____/_/    \_|_____|_|     \____/|_|  \_/_/    \_\ 
                                                      
                                                      
    """
    print(ascii_art)
    print("\n🌿 CaiporaBot - Assistente de Sobrevivência na Amazônia 🌿\n")
    print("Digite sua pergunta ou 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ")
        if pergunta.lower() in ["sair", "exit"]:
            print("\n🌿 Até logo! Lembre-se: na selva, prevenção é tudo!")
            break
        
        resposta = gerar_resposta(pergunta, modelo)
        print(f"\n🌿 CaiporaBot: {resposta}\n")

if __name__ == "__main__":
    iniciar_chat()