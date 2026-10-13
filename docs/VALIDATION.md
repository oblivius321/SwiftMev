# Plano de Validação Científica e Benchmarking — SwiftMev

Para garantir rigor metodológico e fundamentar futuras submissões a periódicos científicos (ex.: *SoftwareX*), as comparações entre métodos clássicos e modelos de inteligência artificial são orientadas por três eixos de avaliação empírica:

```
                  ┌────────────────────────────────────────┐
                  │    DIMENSÕES DE VALIDAÇÃO CIENTÍFICA   │
                  └───────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  [1. Qualidade e Concordância]  [2. Tempo de Execução]     [3. Consumo de Memória]
  • Dice, IoU e erro de contorno • Latência de CPU (ms)     • Pico de alocação RAM
  • Concordância com especialistas• Escalabilidade por MP   • Ausência de vazamento
  • Diâmetro médio e dispersão   • Benchmark reprodutível   • Respeito ao teto de 1 GB
```

## 1. Métricas de Concordância com Imagens Reais
* **Referência humana (Ground Truth)**: Obter anotações manuais independentes de pelo menos dois especialistas (ex.: Química/Biologia celular).
* **Segmentação**:
  * Coeficiente de Similaridade Dice e Índice Jaccard (IoU).
  * Erros médios de distância Hausdorff nos contornos.
* **Migração celular**:
  * Área absoluta livre ($\mu m^2$) e taxa percentual de fechamento frente a referências manuais em séries temporais pareadas ($T_0, T_{12}, T_{24}$).
* **Fibras**:
  * Distribuição dos diâmetros locais (média $\pm$ desvio padrão).
  * Testes de repetibilidade e dispersão em fibras com cruzamento vs. isoladas.

## 2. Protocolo de Benchmark Empírico

O script [benchmarks/run_benchmark.py](file:///c:/Users/Admin/Documents/swiftmev/benchmarks/run_benchmark.py) avalia continuamente o impacto computacional:
* **Resoluções testadas**: $512 \times 512$ (0.26 MP), $1024 \times 1024$ (1.05 MP) e $2048 \times 2048$ (4.19 MP).
* **Pipelines comparados**:
  1. Otsu (global clássico)
  2. Otsu + CLAHE
  3. Sauvola (adaptativo local)
  4. Sauvola + CLAHE
  5. Modo IA (quando modelos ONNX compactos forem integrados)
* **Critério de adoção de IA**: Modelos neurais só justificam sua inclusão se apresentarem ganho estatisticamente significativo em Dice/IoU que compense a sobrecarga de memória e latência em relação aos métodos clássicos otimizados.
