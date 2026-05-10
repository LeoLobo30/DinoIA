# DinoIA

Projeto DinoIA com visao computacional para o `chrome://dino` real e treino por PPO com observacoes numericas de 13 valores.

## Fluxo Principal

O caminho recomendado agora e:

1. Treinar PPO no simulador `DinoEnv`.
2. Continuar o treino PPO no Dino real a partir do checkpoint simulado.
3. Testar o agente PPO puro no Dino real.
4. Usar a heuristica apenas para diagnostico/comparacao.

O fluxo NEAT continua no projeto como legado, mas nao e mais o caminho principal.

## Setup

Crie e ative o ambiente:

```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

Verifique dependencias e artefatos:

```powershell
python -m dinoia doctor
```

## 1. Treinar PPO No Simulador

```powershell
python -m dinoia train-sim-ppo --total-timesteps 250000
```

Esse comando usa `PPO("MlpPolicy", DinoEnv, ...)` com 13 observacoes numericas, sem pixels.
O simulador agora treina em fases, aplica randomizacao de dominio, ruído de observacao, latencia e recompensa auxiliar da heuristica para aproximar o jogo real sem deixar a politica final depender da heuristica.

Artefatos principais:

- `artifacts/ppo/sim_model.zip`
- `artifacts/ppo/best_sim_model.zip`
- `artifacts/ppo/sim_training_config.json`
- `artifacts/ppo/training_history.json`
- `artifacts/ppo/evaluation_history.json`
- `artifacts/ppo/latest_eval.json`
- `artifacts/ppo/latest_summary.txt`

Para validar no simulador:

```powershell
python -m dinoia eval-sim-ppo --best --episodes 10
python -m dinoia play-sim-ppo --best --duration 30
```

O `eval-sim-ppo` e o comando que decide se um modelo merece virar `best_sim_model.zip`.

## 2. Continuar PPO No Dino Real

Abra o Chrome em `chrome://dino`, maximize a janela e rode:

```powershell
python -m dinoia train-real-ppo --total-timesteps 5000 --manual-region 1300 70 1200 400
```

Por padrao, esse comando carrega `artifacts/ppo/best_sim_model.zip` quando existir, continua o treino com `DinoRealEnv` e salva:

- `artifacts/ppo/real_model.zip`
- `artifacts/ppo/best_real_model.zip`
- `artifacts/ppo/real_training_config.json`
- `artifacts/ppo/training_history.json`
- `artifacts/ppo/evaluation_history.json`

## 3. Testar PPO Puro No Dino Real

```powershell
python -m dinoia play-real-ppo --best --manual-region 1300 70 1200 400
```

Esse modo usa apenas a politica PPO para escolher `0=noop`, `1=jump` e `2=duck`.
Nao ha fallback heuristico no comando PPO.

## Diagnostico Visual

Use estes comandos para validar captura, deteccao e controle antes de treinar no jogo real:

```powershell
python -m dinoia diagnose-real --duration 10 --manual-region 1300 70 1200 400
python -m dinoia play-real --manual-region 1300 70 1200 400
```

Sem `--manual-region`, o padrao e `1300 70 1200 400`.

## Resultados

```powershell
python -m dinoia results
```

O resumo prioriza PPO, mostra a avaliacao mais recente e a melhor avaliacao, e ainda exibe NEAT apenas como legado, se houver artefatos antigos.

## Atalhos Da Raiz

```powershell
python main.py
python train_ppo.py
python train_real_ppo.py
python play_ppo.py
python doctor.py
python results.py
python play_real.py
```

Wrappers NEAT legados:

```powershell
python evolve_neat.py
python play_neat.py
```

## Comandos NEAT Legados

Ainda estao disponiveis para comparacao:

```powershell
python -m dinoia evolve-real --generations 20 --population 24 --manual-region 1300 70 1200 400
python -m dinoia play-real-neat --best --manual-region 1300 70 1200 400
python -m dinoia train-sim-parallel --generations 20 --population 16 --workers 4
python -m dinoia play-sim-parallel --best --duration 30
```

## Estrutura

- `dinoia/ppo_agent.py`: treino, fine-tuning, execucao e resumo dos artefatos PPO.
- `dinoia/sim/env.py`: `DinoEnv` Gymnasium simulado, registrado como `Dino-v0`.
- `dinoia/real_env.py`: `DinoRealEnv` Gymnasium real com observacoes numericas.
- `dinoia/real_rl.py`: construcao das 13 observacoes numericas a partir da visao.
- `dinoia/vision.py`: deteccao visual e bounding boxes.
- `dinoia/decision.py`: politica heuristica de diagnostico.
- `dinoia/neat_agent.py`: fluxo NEAT legado.
