# Arquitetura Técnica — SwiftMev

O **SwiftMev** é uma plataforma desktop de análise de imagens científicas desenvolvida para operar com soberania local (100% offline), projetada para computadores de laboratório (hardware de referência: Core i3, 8 GB RAM, sem GPU dedicada).

```
┌──────────────────────────────────────────────────────────────┐
│                    SwiftMev Desktop (C# / .NET)              │
│   • Interface gráfica, importação e visualização             │
│   • Seleção de módulo e calibração micrômetros/pixel         │
│   • Edição interativa de contornos e revisão (planejado)     │
│   • Exportação de laudos, CSV e relatórios                   │
└──────────────────────────────┬───────────────────────────────┘
                               │ Chamada local (Process CLI / JSON)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    SwiftMev Engine (Python)                  │
├──────────────────────────────┬───────────────────────────────┤
│          Modo Rápido         │           Modo IA             │
│  (Filtros Clássicos/Locais)  │      (ONNX Runtime CPU)       │
├──────────────────────────────┼───────────────────────────────┤
│ • OpenCV, NumPy, skimage     │ • TinySAM / MobileSAM ONNX    │
│ • CLAHE opcional             │ • Carregado apenas sob        │
│ • Otsu vs. Sauvola adaptativo│   demanda explícita           │
│ • Morfologia matemática      │ • Sem dependência de GPU      │
├──────────────────────────────┴───────────────────────────────┤
│                    Validação e Resultados                    │
│ • Métricas calibradas (área, fração, diâmetro médio ± std)   │
│ • Imagem PNG com sobreposição de contornos                   │
│ • Auditoria estruturada JSON com benchmark de tempo de CPU   │
└──────────────────────────────────────────────────────────────┘
```

## 1. Contrato da CLI (`cli.py`)

A comunicação entre a interface desktop e o motor científico ocorre via processo isolado, evitando vazamentos de memória na UI.

* **Entrada**:
  * `--module`: `migration` ou `fiber` (obrigatório)
  * `--input`: caminho do arquivo de imagem (obrigatório)
  * `--output`: pasta para gravação dos artefatos (obrigatório)
  * `--um-per-pixel`: fator de escala $\mu m/\text{pixel} > 0$ (obrigatório)
  * `--method`: método de segmentação (`otsu` ou `sauvola`, default: `otsu`)
  * `--clahe`: ativa equalização de contraste local adaptativo (flag booleana)
  * `--invert`: inverte polaridade da máscara binária
  * `--sauvola-window`: tamanho da janela local (default: 35)
  * `--sauvola-k`: coeficiente k de Sauvola (default: 0.2)

* **Saída**:
  * **stdout**: Uma única linha JSON formatada com versão, metadados, métricas e latência (`execution_time_ms`).
  * **stderr**: JSON de erro em caso de falha com exit code 1.
  * **Artefatos em disco**: Imagem de sobreposição PNG e arquivo de métricas JSON.

## 2. Requisitos e Diretrizes de Performance

| Indicador | Meta do SwiftMev |
| :--- | :--- |
| Hardware mínimo | Core i3, 8 GB RAM |
| GPU dedicada | **Não obrigatória** (execução 100% otimizada para CPU) |
| Conexão à internet | **Zero** (100% offline) |
| Consumo de RAM | Meta inferior a 1 GB no modo clássico |
| Modelos de IA | Carregados apenas quando estritamente solicitados |
| RAG / LLM | Módulo adicional desacoplado; não é dependência do núcleo |
