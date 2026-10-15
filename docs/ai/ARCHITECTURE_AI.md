# Arquitetura do Subsistema SwiftMev AI (Alpha 0.6A)

> **Documentação de Engenharia e Rigor Científico**  
> Candidato a Submissão: *SoftwareX* (Elsevier)  
> Plataforma: Local, Offline, Otimizada para CPU (Meta: 8 GB RAM)

---

## 1. Visão Geral

O **SwiftMev AI** expande a plataforma com inferência de redes neurais profundas para segmentação de estruturas microscópicas, começando exclusivamente pela detecção de **nanofibras eletrofiadas em micrografias MEV** (`task="fiber_segmentation"`).

O subsistema é construído sob um princípio inegociável:
> **A IA atua exclusivamente como geradora de máscara binária de segmentação. As medições metrológicas (diâmetros locais por Transformada de Distância euclidiana $L_2$, esqueletização morfológica e desvio padrão) são calculadas de forma determinística pelo motor matemático tradicional (`measurements.py`), sem qualquer interferência de pesos neurais.**

---

## 2. Inferência por Blocos Sobrepostos (*Overlapping Tiling*)

Para preservar detalhes finos de nanofibras (que frequentemente possuem diâmetro de poucos pixels em imagens de $2048 \times 1532$), o SwiftMev AI **não realiza redimensionamento indiscriminado da micrografia**.

### Algoritmo de Tiling Nativo
1. **Recortes Proporcionais:** A imagem é dividida em blocos de dimensão correspondente ao modelo (ex: $256 \times 256$ ou $512 \times 512$), com sobreposição configurável (padrão: $64\text{ px}$).
2. **Janelamento Trapezoidal 2D (*Taper Window*):**
   * Bordas que tocam as extremidades físicas reais da micrografia mantêm peso $1.0$.
   * Zonas internas de sobreposição sofrem decaimento linear suave.
3. **Reconstrução na Resolução Original:** Os mapas de probabilidade de cada bloco são acumulados ponderadamente e normalizados pelo mapa de pesos acumulados:
   $$\text{Prob}(x, y) = \frac{\sum_{i} P_i(x, y) \cdot W_i(x, y)}{\sum_{i} W_i(x, y) + \epsilon}$$
4. **Binarização com Limiar Controlado:** O mapa de probabilidades resultante é binarizado com o limiar definido no manifesto (padrão: $0.5$), gerando uma máscara `uint8 (0 ou 255)` de dimensão idêntica à imagem de entrada.

---

## 3. Contrato de Manifesto de Modelos (`manifest.json`)

Cada modelo local é empacotado em uma subpasta de `models/` e possui um manifesto estruturado com os seguintes campos obrigatórios:

| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `model_id` | `string` | Identificador único do modelo (ex: `test_fiber_onnx_mock`). |
| `name` | `string` | Nome legível para exibição na UI. |
| `task` | `string` | Tarefa estrita do modelo (na Alpha 0.6A: `"fiber_segmentation"`). |
| `architecture` | `string` | Topologia da rede (ex: `MobileNetV3-UNet`, `TinyUNet`). |
| `model_version` | `string` | Versão semântica do modelo. |
| `input_shape` | `list[int]` | Dimensões do tensor de entrada `[1, C, H, W]`. |
| `input_dtype` | `string` | Tipo de dado (`"float32"`). |
| `normalization` | `dict` | Tipo de normalização aplicada (`min_max`, escala, etc.). |
| `output_type` | `string` | `"probability_map"` ou `"logits"`. |
| `threshold` | `float` | Limiar ótimo de binarização ($0.0$ a $1.0$). |
| `license` | `string` | Licença de redistribuição dos pesos (ex: `MIT`). |
| `provenance` | `string` | Origem do treinamento e fontes de dados. |
| `scientific_limitations` | `string` | Restrições de uso e ressalvas metrológicas. |
| `checksum_sha256` | `string` | Hash SHA-256 de 64 caracteres para verificação de integridade. |
| `onnx_filename` | `string` | Nome do arquivo de pesos `.onnx`. |

---

## 4. Códigos de Saída da CLI (`swiftmev.cli`)

A integração preserva a semântica estrita de status:
* **`0`**: Sucesso na execução e geração dos artefatos.
* **`1`**: Erro fatal genérico inesperado.
* **`2`**: Rejeição metrológica científica (`CalibrationError`, `DegenerateMaskError`).
* **`3`**: Erro técnico do subsistema de IA (`AINotAvailableError`, `AIModelNotFoundError`, `TaskIncompatibilityError`, `ChecksumMismatchError`, `TensorMismatchError`).

---

## 5. Limitações Conhecidas (Alpha 0.6A)

1. **Modelo Mock de Teste:** O modelo `test_fiber_onnx_mock` incluído na Alpha 0.6A é uma rede convolucional mínima gerada programmaticamente para validar a infraestrutura de inferência, o tiling e o contrato com a interface C#. **Não deve ser utilizado como segmentador em laudos científicos.**
2. **Exclusividade para Fibras:** Modelos com `task="fiber_segmentation"` são bloqueados para o módulo de migração celular (`migration`), garantindo que redes neurais não sejam aplicadas a domínios não validados.
3. **Imagens Reais de Bancada:** As micrografias contidas na pasta `fotos/` foram catalogadas exclusivamente para leitura de inventário. Nenhum arquivo foi alterado ou utilizado para treino nesta etapa.
