# DinoIA

Refactor profissional do projeto DinoIA em duas frentes:

1. um agente de visao computacional para o `chrome://dino` real;
2. um ambiente simulado proprio, compativel com Gymnasium, para treino com Stable-Baselines3 DQN.

## Visao geral

O projeto foi reorganizado para separar claramente:

- captura de tela;
- percepcao visual;
- decisao da politica;
- controle por teclado;
- simulacao fisica;
- treino e avaliacao de RL.

O que ficou no nucleo do projeto novo e o stack OpenCV + Gymnasium + Stable-Baselines3. Os scripts antigos de OCR/TensorFlow podem ser vistos como legado de prototipo.
Os arquivos legados foram mantidos apenas como referencia historica; o fluxo atual fica em `dinoia/` e nos scripts de entrada da raiz.

## Como rodar

Crie e ative um ambiente virtual local:

```powershell
.\setup_venv.ps1
```

Se quiser GPU NVIDIA:

```powershell
.\setup_venv.ps1 -UseCuda
```

O script termina com um `doctor` automatico para mostrar se o ambiente esta pronto.
Depois ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Fluxo recomendado (simulador)

1. Verifique o ambiente:

```bash
python -m dinoia doctor
```

2. Faca um treino curto de validacao:

```bash
python -m dinoia train --preset quick --device auto --resume auto
```

3. Rode o treino recomendado:

```bash
python -m dinoia train --preset standard --device auto --resume auto
```

4. Veja o resumo dos artefatos:

```bash
python -m dinoia results
```

5. Avalie o melhor modelo salvo:

```bash
python -m dinoia evaluate --latest
```

6. Compare CPU e GPU no seu computador:

```bash
python -m dinoia benchmark --timesteps 3000
```

### Agente real

Abra o Chrome no `chrome://dino`, maximize a janela e deixe o jogo visivel.

```bash
python -m dinoia play-real
```

Se precisar, voce pode informar uma regiao manual:

```bash
python -m dinoia play-real --manual-region 1300 70 1200 400
```

### Jogo e treino juntos

O modo combinado roda o jogo real em modo leve, sem janela de captura, para reduzir travamentos durante o treino.
Importante: esse comando NAO faz treino de RL usando recompensa do jogo real.
Ele roda:
- `play-real` no jogo real;
- `train` no simulador;
em paralelo.

```bash
python -m dinoia play-and-train --preset standard --manual-region 1300 70 1200 400 --timesteps 60000 --device auto
```

Voce pode trocar o tipo de treino com `--preset quick`, `--preset standard` ou `--preset long`.

Se quiser acompanhar a captura, adicione `--show-capture`, mas isso deixa o fluxo mais pesado.

### Rodar o modelo treinado no jogo real

```bash
python -m dinoia play-real-rl --latest --manual-region 1300 70 1200 400 --device auto
```

O fluxo recomendado agora e:

```bash
python -m dinoia play-real-rl --best --manual-region 1300 70 1200 400 --device auto
```

Para inspecionar a captura e a observacao usada pelo RL, sem agir no jogo:

```bash
python -m dinoia diagnose-real --duration 10 --manual-region 1300 70 1200 400
```

Se voce nao passar `--manual-region`, o fluxo real usa por padrao `1300 70 1200 400`.

### Treino no jogo real (DQN real)

Agora existe treino direto no jogo real:

```bash
python -m dinoia train-real --timesteps 5000 --device auto --resume auto --no-debug --manual-region 1300 70 1200 400
```

Esse treino:
- usa captura real da tela;
- executa acoes no teclado real;
- recompensa sobrevivencia e penaliza game over detectado.

Observacoes importantes:
- e mais lento e mais instavel que treino no simulador;
- usa apenas 1 ambiente (`n_envs=1`);
- se interromper com `Ctrl+C`, salva modelo parcial em `artifacts/dqn/dino_real_dqn_interrupted.zip`.

Fluxo pratico recomendado:

1. pre-treinar no simulador

```bash
python -m dinoia train --preset long --device auto --resume auto
```

2. ajustar no real a partir do modelo do simulador

```bash
python -m dinoia train-real --timesteps 5000 --device auto --resume artifacts/dqn/dino_dqn_final.zip --no-debug --manual-region 1300 70 1200 400
```

3. continuar treino real em blocos

```bash
python -m dinoia train-real --timesteps 5000 --device auto --resume artifacts/dqn/dino_real_dqn_final.zip --no-debug --manual-region 1300 70 1200 400
```


Se o Chrome estiver maximizado na tela, voce tambem pode usar:

```bash
python -m dinoia play-real-rl --latest --device auto
```

### Atalhos da raiz

```bash
python main.py
python train_dqn.py
python evaluate_dqn.py
python doctor.py
python results.py
python benchmark.py
```

`python main.py` agora faz o `doctor` inicial do projeto.

## Estrutura

- `dinoia/vision.py`: analise visual e bounding boxes.
- `dinoia/decision.py`: politica heuristica desacoplada da visao.
- `dinoia/control.py`: controle por teclado.
- `dinoia/real_game.py`: loop do agente real.
- `dinoia/real_env.py`: ambiente Gymnasium ligado ao jogo real.
- `dinoia/sim/env.py`: ambiente Gymnasium com fisica e obstaculos.
- `dinoia/rl/train.py`: treino com Stable-Baselines3 DQN.
- `dinoia/rl/train_real.py`: treino DQN usando captura e controle do jogo real.
- `dinoia/rl/evaluate.py`: avaliacao do modelo treinado.

## Decisoes de projeto

- O agente real usa visao computacional com heuristica robusta em vez de OCR.
- O agente de RL usa observacao vetorial para treinar mais rapido e de forma mais estavel.
- O simulador foi pensado para ser simples, deterministico e facil de evoluir.

## Proximos passos naturais

- adicionar um wrapper de registro/env mais completo para multiplos perfis de dificuldade;
- incluir graficos de treino e metricas de avaliacao;
- evoluir o detector visual com tracking temporal mais forte, se quiser maior robustez no jogo real.







