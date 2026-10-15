# SwiftMev AI — Relatório de Prontidão para Treinamento (Alpha 0.6C.1)

**Data:** Outubro de 2026  
**Status da Auditoria:** Concluída  
**Classificação de Prontidão do Dataset:** `NOT_READY` (Não preparado para treinamento supervisionado)  
**Ambiente Host Analisado:** Intel Core i5-4590 @ 3.30 GHz (4 cores / 4 threads), 16 GB RAM, Windows 10/11  

---

## 1. Sumário Executivo e Veredito

> [!CAUTION]
> **Classificação Final:** **`NOT_READY`**  
> Embora o motor de inferência ONNX (Alpha 0.6A) e o Dataset Studio (Alpha 0.6B) estejam 100% operacionais e testados (76/76 testes aprovados), **o repositório atual possui exatamente 0 (zero) máscaras de segmentação aprovadas ou revisadas por humanos para as 85 micrografias MEV reais.**  
>
> Iniciar o treinamento de redes neurais supervisionadas neste momento exigiria treinar a rede sobre máscaras geradas automaticamente por Otsu ou Sauvola. Isso configuraria **vício metodológico circular** (a rede aprenderia a replicar os artefatos dos próprios filtros clássicos) e violaria o compromisso de rigor científico do SwiftMev.  
>
> **Recomendação:** Concluir a fase Alpha 0.6C.1 com a validação das anotações humanas e confirmação dos cadernos laboratoriais antes de disparar o treinamento supervisionado (Alpha 0.6C.2).

---

## 2. Auditoria Detalhada do Dataset e Anotações

### 2.1. Censo de Arquivos e Micrografias Elegíveis

A varredura exaustiva do repositório `fotos/` através do módulo `swiftmev.studio.inventory` estabelece a base física real:

| Item | Quantidade | Descrição Metrológica |
| :--- | :---: | :--- |
| **Total de Arquivos em `fotos/`** | **87** | Reconciliação completa do diretório. |
| **Micrografias TIFF MEV Elegíveis** | **85** | Imagens brutas TIFF, $2048 \times 1532$ pixels, 8-bit, 3 canais. |
| **Arquivos de Relatório Auxiliares** | **2** | `getIT_1_relatorio.csv` (313 B) e `getIT_1_relatorio.png` (2.9 MB). |
| **Máscaras de Referência Aprovadas (`approved`)** | **0** | Nenhuma micrografia real possui Ground Truth aprovado. |
| **Máscaras de Referência Revisadas (`reviewed`)** | **0** | Nenhuma micrografia real passou por revisão formal. |
| **Máscaras em Rascunho / Preliminares (`draft`/`annotated`)** | **0** | Nenhuma anotação persistida no repositório `fotos/`. |
| **Amostras Físicas Confirmadas (`physical_sample_id`)** | **0** | Todas as 85 micrografias estão como `unknown` (pendente de revisão). |
| **Grupos Conservadores Provisórios (`conservative_group_id`)** | **20** | Clusters derivados da árvore de pastas para bloqueio de vazamento. |

### 2.2. Distribuição por Formulação Química Nominal

As 85 micrografias MEV estão distribuídas nas seguintes pastas de formulação:

| Formulação Nominal | Micrografias | Grupos Conservadores | Observações de Catalogação |
| :--- | :---: | :---: | :--- |
| **`5_`** | 25 | 6 | Contém pastas `100-0`, `40-60`, `60-40`, `80-20` e `Control/5_`. |
| **`10_`** | 20 | 5 | Contém pastas `100-0`, `40-60`, `60-40`, `80-20` e `Control/10_`. |
| **`15_`** | 20 | 5 | Contém pastas `100-0`, `40-60`, `60-40`, `80-20` e `Control/15_`. |
| **`20_`** | 20 | 4 | Contém pastas `100-0`, `40-60`, `60-40`, `80-20` e `Control/20_`. |
| **Total Geral** | **85** | **20** | **100% na resolução de $2048 \times 1532$ px.** |

### 2.3. Duplicatas Criptográficas Críticas Detectadas

O motor de integridade por SHA-256 identificou **4 pares de arquivos bit-a-bit idênticos** entre formulações nominais diferentes:

1. `5_/40-60/getIT_53.tif` $\longleftrightarrow$ `10_/40-60/getIT_53.tif` (`SHA-256: 43926cb17e...`)
2. `5_/40-60/getIT_54.tif` $\longleftrightarrow$ `10_/40-60/getIT_54.tif` (`SHA-256: 184518743d...`)
3. `5_/40-60/getIT_55.tif` $\longleftrightarrow$ `10_/40-60/getIT_55.tif` (`SHA-256: c3e6e3c04d...`)
4. `5_/40-60/getIT_56.tif` $\longleftrightarrow$ `10_/40-60/getIT_56.tif` (`SHA-256: 673d325a77...`)

*Implicação Científica:* Uma divisão aleatória ingênua alocaria essas imagens entre treino e teste, causando vazamento catastrófico de dados. O algoritmo `partition_groups_disjoint` agrupa esses pares no mesmo cluster compulsoriamente.

### 2.4. Disponibilidade e Validade de Calibrações

- **84 das 85 micrografias (98.8%)** não possuem metadados de calibração micrométrica ($\mu\text{m/px}$) embutidos em tags TIFF nem em arquivos vinculados.
- **1 único arquivo (`5_/100-0/getIT_1_relatorio.csv`)** cita um valor pontual de `0.008202 µm/px` referente a uma execução anterior.
- **Diretriz Metrológica Adotada:** Em conformidade com o princípio de isolamento de calibração, esse valor **não foi nem será propagado** para as outras 84 imagens. Operações de segmentação e rotulagem funcionam em coordenadas de pixel; conversões para micrômetros permanecem bloqueadas com `CalibrationError`.

---

## 3. Problemas que Impedem o Treinamento Imediato

1. **Ausência de Rótulos Humanos de Referência:** Não há dados anotados supervisionados para alimentar a função de perda (Loss).
2. **Independência Experimental Desconhecida:** A ausência de identificação do lote de síntese e da amostra física impede garantir que o conjunto de teste represente generalização biológica/química real, e não apenas interpolação de uma mesma mecha de polímero.
3. **Ambiente PyTorch:** O ambiente de produção atual possui apenas `onnxruntime` e bibliotecas clássicas de visão computacional. O PyTorch ainda precisa ser provisionado de forma isolada em `requirements-train.txt`.

---

## 4. Proposta da Arquitetura Neural: CompactFiberUNet

Para atender ao hardware-alvo (CPU Intel Haswell i5-4590, 8–16 GB RAM) e garantir execução ágil offline sem GPU dedicada, propõe-se uma U-Net compacta e eficiente.

```
                              CompactFiberUNet
                   (1 canal entrada, 256x256 patch)
                                 │
                   ┌─────────────┴─────────────┐
                   │ Conv 1->32, ReLU, BN      │ ──[Skip 1]──┐
                   │ Conv 32->32, MaxPool (2x) │             │
                   └─────────────┬─────────────┘             │
                                 │                           │
                   ┌─────────────┴─────────────┐             │
                   │ Conv 32->64, ReLU, BN     │ ──[Skip 2]┐ │
                   │ Conv 64->64, MaxPool (2x) │           │ │
                   └─────────────┬─────────────┘           │ │
                                 │                         │ │
                   ┌─────────────┴─────────────┐           │ │
                   │ Conv 64->128, ReLU, BN    │ ──[Skip 3] │
                   │ Conv 128->128, MaxPool    │         │ │ │
                   └─────────────┬─────────────┘         │ │ │
                                 │                       │ │ │
                   ┌─────────────┴─────────────┐         │ │ │
                   │ BOTTLENECK (128->256)     │         │ │ │
                   │ Dropout(p=0.2), ReLU, BN  │         │ │ │
                   └─────────────┬─────────────┘         │ │ │
                                 │                       │ │ │
                   ┌─────────────┴─────────────┐         │ │ │
                   │ UpConv(256->128) + Concat │ <───────┘ │ │
                   │ Conv 256->128, ReLU, BN   │           │ │
                   └─────────────┬─────────────┘           │ │
                                 │                         │ │
                   ┌─────────────┴─────────────┐           │ │
                   │ UpConv(128->64) + Concat  │ <─────────┘ │
                   │ Conv 128->64, ReLU, BN    │             │
                   └─────────────┬─────────────┘             │
                                 │                           │
                   ┌─────────────┴─────────────┐             │
                   │ UpConv(64->32) + Concat   │ <───────────┘
                   │ Conv 64->32, ReLU, BN     │
                   └─────────────┬─────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │ Conv 32->1 (1x1), Logits  │
                   └───────────────────────────┘
```

### Especificações Técnicas:

- **Parâmetros Totais:** ~1.950.000 parâmetros (1.95 M) contra 31.000.000 (31 M) de uma U-Net padrão de Ronneberger.
- **Entrada:** `[B, 1, 256, 256]`, ponto flutuante normalizado em $[0, 1]$.
- **Saída:** `[B, 1, 256, 256]`, mapa de probabilidades após ativação Sigmoid.
- **Normalização:** `BatchNorm2d` (ou `GroupNorm(8, C)` para batch size reduzido em CPU).
- **Prevenção de Overfitting:** Camada `Dropout(0.2)` no gargalo (bottleneck) e penalização $L_2$ (`weight_decay = 1e-4`).
- **Tiling para Inferência Panorâmica:**
  Processamento por blocos de $256 \times 256$ com sobreposição de $32\text{ px}$ (stride 224) e fusão por ponderação de janela 2D (Hann Window / Gaussian Weighting), garantindo ausência de descontinuidades nas bordas dos blocos.

---

## 5. Estratégia de Treinamento

### 5.1. Função de Perda Composta (Loss Híbrida)

Devido ao desbalanceamento intrínseco (fibras finas ocupam entre 10% e 30% da micrografia):

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCEWithLogits}} + 0.5 \cdot \mathcal{L}_{\text{SoftDice}}$$

Onde a Soft Dice Loss é formulada como:

$$\mathcal{L}_{\text{SoftDice}} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}$$

### 5.2. Hiperparâmetros e Otimização

- **Otimizador:** AdamW ($\beta_1=0.9, \beta_2=0.999$, $\text{weight\_decay}=10^{-4}$).
- **Taxa de Aprendizado:** $\text{lr} = 3 \times 10^{-4}$ com decaimento por cosseno (`CosineAnnealingLR`, $\text{T\_max}=50, \text{eta\_min}=10^{-6}$).
- **Tamanho do Lote (Batch Size):** 8 ou 16 patches de $256 \times 256$.
- **Épocas:** 50 épocas máximas.
- **Early Stopping:** Interrupção se a métrica *Validation Dice* não melhorar por 10 épocas consecutivas.
- **Reprodutibilidade Estrita:** Fixação de sementes (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `torch.backends.cudnn.deterministic = True`).

### 5.3. Aumento de Dados (Data Augmentation)

Aplicado **estritamente e unicamente ao conjunto de Treinamento**:
- Inversão horizontal aleatória ($p=0.5$).
- Inversão vertical aleatória ($p=0.5$).
- Rotação ortogonal aleatória ($90^\circ, 180^\circ, 270^\circ$).
- Ajuste moderado de brilho e contraste ($\pm 15\%$).
- Adição sutil de ruído Gaussiano ($\sigma \le 0.02$).

> [!IMPORTANT]
> Os conjuntos de Validação e Teste serão avaliados em imagens cruas, sem qualquer aumento sintético de dados.

---

## 6. Estratégia de Validação Científica Comparativa

O experimento comparará três abordagens sob as mesmas condições:

| Módulo | Tipo | Entrada | Complexidade |
| :--- | :--- | :--- | :--- |
| **Otsu** | Limiarização Global Clássica | Monocromática | $O(N)$ em CPU |
| **Sauvola** | Limiarização Adaptativa Local | Monocromática ($w=35, k=0.2$) | $O(N)$ com integral |
| **CompactFiberUNet** | Rede Neural Convolucional | Monocromática ($256 \times 256$ tiles) | $O(N)$ convolucional |

### Métricas de Avaliação Rigorosa:

1. **Métricas de Segmentação Pixel-a-Pixel:**
   - **Dice Score (F1):** Coincidência entre máscara predita e máscara de referência.
   - **IoU (Jaccard Index):** Interseção sobre união.
   - **Precision e Recall:** Controle de falsos positivos (fundo como fibra) e falsos negativos (fibras omitidas).
2. **Métricas Metrológicas de Diâmetro:**
   - Erro Médio Absoluto do Diâmetro: $\text{MAE} = \frac{1}{M} \sum |\bar{d}_{\text{pred}} - \bar{d}_{\text{gt}}| \quad (\mu\text{m})$.
   - Desvio Padrão do Diâmetro: Comparação da dispersão estatística de calibração.
3. **Agrupamento por Amostra Independente:**
   Resultados reportados não apenas globalmente, mas particionados por grupo químico (`5%`, `10%`, `15%`, `20%`) e por micrografia individual.
4. **Benchmark com Imagens Sintéticas Calibradas:**
   Avaliação prévia contra os padrões sintéticos de `datasets/benchmark_v02/` (fibras finas de 6px, médias de 16px, espessas de 30px e ruidosas) cuja verdade de solo geométrica é analítica.

---

## 7. Estimativa de Recursos e Desempenho (Hardware Host)

*As estimativas abaixo foram calculadas com base nas especificações do processador Intel Core i5-4590 (4 núcleos / 4 threads, Haswell) com 16 GB RAM executando Windows 10/11:*

| Operação | Consumo de Memória RAM (Estimado) | Tempo de Execução (Estimado) |
| :--- | :---: | :---: |
| **Extração de Patches ($256 \times 256$)** | $\sim 350\text{ MB}$ | $< 10\text{ s}$ |
| **Treinamento por Época (CPU, 500 patches)** | $\sim 1.8\text{ GB} - 2.2\text{ GB}$ | $\sim 45\text{ s} - 65\text{ s}$ por época |
| **Treinamento Completo (50 épocas)** | Pico $< 2.5\text{ GB}$ | $\sim 35\text{ min} - 55\text{ min}$ |
| **Exportação e Validação ONNX** | $\sim 400\text{ MB}$ | $< 5\text{ s}$ |
| **Inferência de Micrografia Completa ($2048 \times 1532$)** | Incremento $< 150\text{ MB}$ | $\sim 1.2\text{ s} - 1.8\text{ s}$ (4 threads) |

> [!NOTE]
> Os valores acima são estimativas nominais e serão validados empiricamente com medições via `psutil` e benchmarks cronometrados na Alpha 0.6C.2.

---

## 8. Auditoria de Funcionalidades do Dataset Studio (Alpha 0.6B)

A auditoria de conformidade das funcionalidades do Dataset Studio implementadas na Alpha 0.6B apresentou os seguintes resultados:

1. **Persistência de Anotações:** `save_annotation_bundle` salva corretamente pares de máscara PNG de 8-bit sem perdas e arquivos sidecar JSON (`*_ann.json`) contendo histórico versionado de autores e revisores.
2. **Prevenção de Sobrescrita:** A CLI e a interface desktop agora bloqueiam sobrescrita silenciosa de releases. A interface desktop exibe um diálogo de confirmação explícito caso o usuário tente exportar sobre uma versão pré-existente.
3. **Isolamento de Origem:** As micrografias em `fotos/` são acessadas em modo estritamente somente-leitura. O `.gitignore` foi configurado para impedir inclusão acidental de micrografias de pesquisa no repositório público.
4. **Conexão na Interface Desktop WPF:** A aba **"📦 Dataset Studio"** está plenamente vinculada à CLI por chamadas assíncronas assépticas com captura de `stdout`/`stderr` em JSON, comunicando-se com o canvas de visualização e edição de máscaras.
5. **Suíte de Testes:** **76 de 76 testes aprovados** sem regressões nas ferramentas clássicas (Otsu e Sauvola).

---

## 9. Plano de Implementação para a Alpha 0.6C.2 (Pós-Aprovação)

Quando o processo de rotulagem das primeiras máscaras for concluído pelo pesquisador e aprovado, a implementação técnica seguirá o seguinte plano de arquivos:

| Arquivo Planejado | Finalidade Técnica |
| :--- | :--- |
| `engine/requirements-train.txt` | Dependências exclusivas de treino (`torch>=2.2`, `torchvision>=0.17`). |
| `engine/swiftmev/ai/unet.py` | Implementação PyTorch da arquitetura `CompactFiberUNet`. |
| `engine/swiftmev/ai/dataset.py` | `Dataset` e `DataLoader` PyTorch com extração de patches e aumento de dados. |
| `engine/swiftmev/ai/losses.py` | Implementação da função de perda composta BCE + Soft Dice. |
| `engine/swiftmev/ai/trainer.py` | Laço de treinamento, validação, early stopping e exportação automática para ONNX. |
| `engine/swiftmev/ai/evaluator.py` | Comparador automatizado Otsu vs Sauvola vs U-Net com métricas de diâmetro. |
| `tests/test_training_pipeline.py` | Testes automatizados do pipeline de treino com dados sintéticos e validação de convergência. |

---

## 10. Conclusão

O SwiftMev possui agora toda a infraestrutura de engenharia e curadoria necessária para treinar redes neurais profundas de alta fidelidade científica. O impedimento atual é puramente metrológico: **a necessidade de Ground Truth humano anotado e revisado.**

Assim que os primeiros lotes de máscaras forem aprovados através da aba **"📦 Dataset Studio"**, a Alpha 0.6C.2 poderá ser iniciada para treinar o primeiro modelo ONNX real de segmentação de nanofibras.
