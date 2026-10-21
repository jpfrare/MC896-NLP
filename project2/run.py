#!/usr/bin/env python3
"""Ponto de entrada único do Projeto 2: cria o ambiente, instala as dependências, roda tudo e
abre o painel de resultados.

    python3 run.py                 # tudo (na 1ª vez cria .venv e instala; depois só roda)
    python3 run.py --rapido        # pula a ablação de n-gramas (a etapa mais lenta)
    python3 run.py --reinstalar    # apaga e recria o .venv do zero
    python3 run.py --nao-abrir     # não abre o navegador no final

Funciona em Linux, macOS e Windows, com qualquer Python 3.8+ para rodar ESTE script.
O ambiente do projeto (project2/.venv) usa Python 3.12, porque o modelo en_core_sci_sm 0.5.4
exige spaCy 3.7, que só existe até o Python 3.12. Para obter o Python 3.12 o script usa,
nesta ordem:
  1. uv (recomendado; baixa o Python 3.12 sozinho):  https://docs.astral.sh/uv/
  2. um interpretador python3.12 já instalado no sistema.
Precisa de internet na primeira execução (pacotes e modelos spaCy).
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS = ROOT / "requirements.txt"
STAMP = VENV / ".requirements.sha256"
DASHBOARD = ROOT / "assets" / "html" / "index.html"
PYVER = "3.12"


def say(msg):
    print(f"\n>>> {msg}", flush=True)


def run(cmd, **kw):
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def venv_version():
    py = venv_python()
    if not py.exists():
        return None
    out = subprocess.run([str(py), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def find_uv():
    uv = shutil.which("uv")
    if uv:
        return [uv]
    probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True)
    return [sys.executable, "-m", "uv"] if probe.returncode == 0 else None


def create_venv(uv):
    if uv:
        run(uv + ["venv", "--python", PYVER, str(VENV)])
        return
    py312 = shutil.which(f"python{PYVER}") or (shutil.which("py") and f"py -{PYVER}")
    if not py312:
        sys.exit(
            f"\nNão encontrei o uv nem um Python {PYVER}. Instale um dos dois e rode de novo:\n"
            "  uv:          Arch: sudo pacman -S uv · macOS: brew install uv · outros: https://docs.astral.sh/uv/\n"
            f"  Python {PYVER}: Arch: yay -S python312 · Ubuntu: sudo apt install python{PYVER}-venv\n")
    run(py312.split() + ["-m", "venv", str(VENV)])
    run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"])


def requirements_hash() -> str:
    return hashlib.sha256(REQS.read_bytes()).hexdigest()


def install(uv):
    if uv:
        run(uv + ["pip", "install", "--python", str(venv_python()), "-r", str(REQS)])
        run(uv + ["pip", "check", "--python", str(venv_python())])
    else:
        run([venv_python(), "-m", "pip", "install", "-r", str(REQS)])
        run([venv_python(), "-m", "pip", "check"])
    STAMP.write_text(requirements_hash())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="pula a ablação de n-gramas")
    ap.add_argument("--reinstalar", action="store_true", help="recria o ambiente do zero")
    ap.add_argument("--nao-abrir", action="store_true", help="não abre o painel no navegador")
    args = ap.parse_args()

    uv = find_uv()
    if args.reinstalar and VENV.exists():
        say("Apagando o ambiente antigo")
        shutil.rmtree(VENV)
    version = venv_version()
    if version != PYVER:
        if VENV.exists():
            say(f"O .venv existente usa Python {version or '?'}; recriando com Python {PYVER}")
            shutil.rmtree(VENV)
        say(f"Criando o ambiente virtual (Python {PYVER}) em {VENV}")
        create_venv(uv)
    if not STAMP.exists() or STAMP.read_text().strip() != requirements_hash():
        say("Instalando as dependências (requirements.txt)")
        install(uv)
    else:
        say("Dependências já instaladas (requirements.txt não mudou)")

    say("Rodando o sistema de comparação (src/run_all.py)")
    cmd = [venv_python(), "-m", "src.run_all"] + (["--rapido"] if args.rapido else [])
    env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg")
    status = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env).returncode

    if DASHBOARD.exists():
        say(f"Painel: {DASHBOARD}")
        if not args.nao_abrir:
            webbrowser.open(DASHBOARD.as_uri())
    if status:
        print("\nAlgumas etapas falharam (veja o resumo acima); o painel e o README mostram o aviso.")
    sys.exit(status)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"\nO comando acima falhou (código {exc.returncode}). Confira a conexão com a internet e tente "
                 "`python3 run.py --reinstalar`. Se persistir, envie a saída completa para a equipe.")
    except KeyboardInterrupt:
        sys.exit("\nInterrompido.")
