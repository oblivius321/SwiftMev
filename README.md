# SwiftMev — Análise Científica de Microscopia e MEV (Alpha 0.4)

> **SoftwareX (Elsevier) Submission Candidate**  
> Plataforma científica local de alta performance para processamento de imagens de Microscopia Eletrônica de Varredura (MEV) e ensaios *in vitro*, combinando interface desktop C# WPF (.NET 8) com motor científico em Python.

---

## 🔬 Visão Geral

O **SwiftMev** é projetado para operar localmente com rigor metrológico em computadores de laboratório (sem exigir GPU ou conexão à internet). O sistema contempla:

1. **Diâmetro e Morfometria de Fibras**: extração de diâmetro local por Transformada de Distância euclidiana combinada com esqueletização morfológica, com auditoria de viés de discretização.
2. **Migração Celular (Scratch Assay)**: quantificação de fração de área livre e delimitação de bordas celulares.
3. **Calibração Espacial Interativa**: calibração obrigatória por valor numérico ou via ferramenta de régua gráfica interativa sobre a barra de escala do MEV.
4. **Editor Manual de Máscaras**: ferramentas de pincel e borracha em tempo real para curadoria de segmentação, preservando a imagem original imutável e recalculando métricas diretamente da máscara editada.
5. **Projetos Persistentes e Exportação**: formato de projeto `.json` com auditoria completa de parâmetros e exportação de imagens de alta resolução com barra de escala física gravada.

---

## 🏛️ Arquitetura do Sistema

```
swiftmev/
├── apps/
│   └── desktop/                  # Interface gráfica Windows em C# / WPF (.NET 8)
│       ├── SwiftMev.Desktop.csproj  # Projeto configurado com AllowUnsafeBlocks para manipulação de pixels
│       ├── MainWindow.xaml       # Visualizador microscópico com Pan/Zoom, camadas e réguas
│       └── MainWindow.xaml.cs    # Motor interativo, edição manual e integração com swiftmev.cli
├── engine/                       # Motor científico em Python (v0.3.0)
│   ├── swiftmev/                 # Pacote modular
│   │   ├── imaging.py            # Conversão segura de tipos e pré-processamento CLAHE
│   │   ├── segmentation.py       # Segmentação Otsu e Sauvola
│   │   ├── measurements.py       # Medições a partir de máscara pré-computada
│   │   ├── validation.py         # Bloqueio obrigatório (CalibrationError, DegenerateMaskError)
│   │   ├── pipeline.py           # Orquestração (auto e a partir de máscara editada)
│   │   └── cli.py                # Interface CLI com suporte a --mask e exit code 2
│   └── cellscope/                # Shim de compatibilidade com DeprecationWarning
├── benchmarks/                   # Bateria de validação quantitativa e gerador estocástico
│   ├── geometric_generator.py    # Gerador com jitter estocástico (centro, ângulo, comprimento)
│   ├── benchmark_alpha03.py      # Benchmark completo com amostragem contínua de RSS
│   └── rss_monitor.py            # Monitoramento contínuo de consumo de memória
├── tests/                        # 37 testes automatizados (34 unitários + 3 de integração E2E)
└── outputs/                      # Diretório de exportação de métricas JSON, relatórios e máscaras
```

---

## 📊 Critérios de Aceitação da Alpha 0.4 (Checklist 9/9 Concluído)

- [x] **1. Compilação e Execução**: Aplicativo C# compila com 0 erros via .NET 8 SDK (`SwiftMev.exe`).
- [x] **2. Importação Amigável**: Pesquisador seleciona arquivos via diálogo padrão sem terminal.
- [x] **3. Visualizador Microscópico**: Zoom contínuo na roda do mouse (0.05× a 40×), Pan com arrasto e indicador de posição $(X, Y)$ em pixels.
- [x] **4. Sobreposição de Máscara**: Visualização de contornos e máscara colorida com controle de opacidade (Alpha Blending 0 a 100%).
- [x] **5. Calibração Obrigatória**: Bloqueio de cálculos se escala não for definida; ferramenta de régua interativa sobre a barra de escala do MEV com conversão automática para $\mu\text{m}/\text{px}$.
- [x] **6. Edição Manual de Máscaras**: Pincel (+) e borracha (-) com raio ajustável e botão para recalcular métricas a partir da máscara editada (`--mask`).
- [x] **7. Projetos Persistentes**: Salva e reabre projetos com imagem original, máscara e parâmetros.
- [x] **8. Relatórios de Auditoria**: Exportação de CSV estruturado e imagem anotada com barra de escala física gravada.
- [x] **9. Operação 100% Offline**: Motor local autônomo, sem dependências de nuvem.

---

## 🚀 Como Executar

### 1. Testes Automatizados (Python)

```bash
# Executar a suíte de 37 testes (unitários + integração)
pytest -v
```

### 2. Compilar e Executar o Aplicativo Desktop (C# WPF)

```bash
# Compilar projeto
dotnet build apps/desktop/SwiftMev.Desktop.csproj

# Executar binário
.\apps\desktop\bin\Debug\net8.0-windows\SwiftMev.exe
```

### 3. Linha de Comando (CLI)

```bash
# Análise automática
python -m swiftmev.cli --module fiber --input datasets/samples/fiber_sample.png --output outputs --um-per-pixel 0.5 --save-mask

# Recálculo a partir de máscara editada manualmente
python -m swiftmev.cli --module fiber --input datasets/samples/fiber_sample.png --mask outputs/temp_edited_mask.png --output outputs --um-per-pixel 0.5
```

---

## 📜 Licença

Distribuído sob licença MIT. Consulte `LICENSE` para mais detalhes.
