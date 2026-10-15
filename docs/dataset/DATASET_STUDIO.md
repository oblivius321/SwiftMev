# SwiftMev AI — Scientific Dataset Studio (Alpha 0.6B)

Manual de Arquitetura, Procedimentos Científicos e Guia de Uso.

---

## 1. Visão Geral e Filosofia Científica

O **Dataset Studio** é o subsistema do SwiftMev responsável pela catalogação, anotação com ciclo de vida rigoroso, validação de integridade e exportação reprodutível de dados para visão computacional científica em micrografias MEV (Microscopia Eletrônica de Varredura).

### Princípios Metodológicos

1. **Separação Taxonômica Rigorosa:** Não inferir independência de amostras nem grupos físicos a partir de nomes de pastas de sistema de arquivos. A estrutura de diretórios é tratada como agrupamento conservador provisório (`conservative_group_id`), enquanto a amostra física real permanece `unknown` ou `requires_review` até confirmação experimental explícita.
2. **Prevenção Disjunta de Vazamento (Pairwise Disjointness):**
   A condição tripla clássica $Train \cap Val \cap Test = \emptyset$ é matematicamente insuficiente para impedir contaminação. O Dataset Studio exige disjunção estrita aos pares:
   $$\text{Train} \cap \text{Val} = \emptyset \quad\land\quad \text{Train} \cap \text{Test} = \emptyset \quad\land\quad \text{Val} \cap \text{Test} = \emptyset$$
   Nenhuma duplicata criptográfica (SHA-256) ou perceptual (dHash $\le 2$ bits) e nenhum grupo experimental pode atravessar as partições de treino, validação e teste.
3. **Ciclo de Vida de Anotação em 4 Estados:**
   $$\text{draft} \longrightarrow \text{annotated} \longrightarrow \text{reviewed} \longrightarrow \text{approved}$$
   Máscaras automáticas (Otsu, Sauvola, IA prévia) são registradas com proveniência explícita e **não podem ser promovidas diretamente a Ground Truth Aprovado** sem inspeção humana e revisão documental.
4. **Isolamento de Calibração Metrológica:**
   A existência de fatores de calibração em relatórios pontuais (ex: `0.008202 µm/px` em `getIT_1_relatorio.csv`) não autoriza a propagação tácita para o restante das micrografias. Imagens sem calibração documentada operam normalmente para anotação e segmentação em pixels, mas medições físicas em micrômetros são bloqueadas com `CalibrationError`.
5. **Garantias Declaradas com Honestidade Científica:**
   O software emite os campos `automated_leakage_checks_passed` e `experimental_independence_reviewed` separadamente, abstendo-se de afirmações absolutas de "zero leakage" sem validação presencial das bancadas.

---

## 2. Reconciliação do Inventário (`fotos/`)

Durante a auditoria da Alpha 0.6B, foi reconciliada a contagem do repositório `fotos/`:

| Categoria | Quantidade | Observações |
| :--- | :--- | :--- |
| **Micrografias TIFF MEV** | **85** | Arquivos `.tif` originais de $2048 \times 1532$, 8-bit, 3 canais. |
| **Relatórios e Gráficos Auxiliares** | **2** | `getIT_1_relatorio.csv` e `getIT_1_relatorio.png` (em `5_/100-0/`). |
| **Total de Arquivos em `fotos/`** | **87** | Reconciliação definitiva 100% reproduzível. |

### Grupos de Duplicatas Exatas Detectados (SHA-256)

A varredura criptográfica revelou 4 pares de imagens idênticas cruzando formulações nominais distintas:

1. `10_/40-60/getIT_53.tif` $\longleftrightarrow$ `5_/40-60/getIT_53.tif`
2. `10_/40-60/getIT_54.tif` $\longleftrightarrow$ `5_/40-60/getIT_54.tif`
3. `10_/40-60/getIT_55.tif` $\longleftrightarrow$ `5_/40-60/getIT_55.tif`
4. `10_/40-60/getIT_56.tif` $\longleftrightarrow$ `5_/40-60/getIT_56.tif`

O particionador `partition_groups_disjoint` agrupa essas imagens compulsoriamente no mesmo split para impedir que réplicas exatas contaminem o conjunto de validação ou teste.

---

## 3. Assistente de Controle de Qualidade (QC)

O módulo de QC (`annotations.py`) atua como inspetor consultivo (não-destrutivo) fornecendo alertas sobre problemas típicos de anotação de nanofibras:

- `TINY_COMPONENTS`: Sinaliza fragmentos minúsculos desconectados ($< 15\text{ px}$), comuns em ruído de limiarização.
- `IRREGULAR_BORDERS`: Sinaliza contornos com circularidade fractal/serrilhada excessiva.
- `SUSPICIOUSLY_LOW_COVERAGE` / `SUSPICIOUSLY_HIGH_COVERAGE`: Avisa se a densidade de primeiro plano estiver abaixo de 2% ou acima de 85%.
- `EMPTY_MASK`: Alerta se nenhum pixel tiver sido anotado.
- `UNANNOTATED_BRIGHT_REGIONS`: Detecta feixes de alta intensidade na micrografia original que foram excluídos da máscara.

---

## 4. Exemplo de Manifesto de Release (`manifest.json`)

```json
{
  "dataset_name": "swiftmev_fiber_dataset",
  "dataset_version": "0.1.0",
  "created_at": "2026-10-14T21:00:00Z",
  "generator": "SwiftMev Dataset Studio",
  "total_micrographs": 85,
  "total_annotations": 85,
  "splits_count": {
    "train": 59,
    "val": 13,
    "test": 13
  },
  "automated_leakage_checks_passed": true,
  "experimental_independence_reviewed": false,
  "integrity_hashes": {
    "images/MG_10___40-60__getIT_53_tif.tif": "43926cb17eebc558c734493010b9a957b486a4e3218aa0eb064d18ecad354d19",
    "masks/MG_10___40-60__getIT_53_tif_mask.png": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "annotations/MG_10___40-60__getIT_53_tif_ann.json": "8a329ef89b78294...",
    "splits/train.txt": "f7842c...",
    "inventory.csv": "c138da..."
  },
  "scientific_notes": "Particionamento por grupos conservadores e agrupamento forçado de pares idênticos.",
  "limitations": "Identificação de amostra física dependente de confirmação com a equipe de síntese laboratorial."
}
```

---

## 5. Procedimento de Uso Manual

### A. Pela Interface Gráfica Desktop (C# WPF)

1. **Abrir a Aplicação:** Execute `SwiftMev.exe` ou inicie via terminal com `dotnet run --project apps/desktop/SwiftMev.Desktop.csproj`.
2. **Acessar o Dataset Studio:** No painel lateral esquerdo, clique na aba **"📦 Dataset Studio"**.
3. **Escanear Micrografias:**
   - No campo de pasta raiz, informe `fotos` (ou o caminho absoluto desejado).
   - Clique em **"🔍 Escanear"**.
   - O sistema calculará os hashes criptográficos SHA-256 e perceptuais dHash, listando as 85 micrografias e destacando alertas de duplicatas.
4. **Inspecionar e Comparar:**
   - Clique em qualquer micrografia da lista para carregá-la imediatamente no visualizador central.
   - Use o slider de **Opacidade da Máscara** e os checkboxes de **Camadas (Contornos, Máscara)** na barra de ferramentas do canvas para inspecionar a sobreposição.
5. **Anotar e Executar QC:**
   - Use os pincéis da aba **"🖌️ Edição"** para retocar a máscara.
   - Retorne à aba **"📦 Dataset Studio"** e clique em **"🛡️ Assistente de Qualidade (QC)"** para checar fragmentos ou desvios de borda.
6. **Gerenciar o Ciclo de Vida:**
   - Clique em **"💾 Salvar Rascunho"** para salvar um rascunho (`draft`).
   - Clique em **"📋 Submeter Anotação"** quando terminar (`annotated`).
   - Um revisor clica em **"🔍 Marcar Revisado"** (`reviewed`).
   - Após validação independente, clique em **"⭐ Aprovar Ground Truth"** (`approved`).
7. **Calcular Splits e Exportar:**
   - Clique em **"⚖️ Calcular Splits"** para validar disjunção aos pares sem vazamento.
   - Clique em **"🚀 Exportar Release"** para gerar a pasta `dataset_releases/` com manifesto e hashes.

### B. Pela Linha de Comando (CLI / Automação)

```powershell
# Definir PYTHONPATH
$env:PYTHONPATH="engine"

# 1. Escanear inventário e integridade
python -m swiftmev.cli --studio-scan fotos

# 2. Calcular particionamento disjunto e sem vazamento
python -m swiftmev.cli --studio-split fotos

# 3. Executar inspeção de controle de qualidade (QC) em máscara
python -m swiftmev.cli --studio-qc --mask caminho_para_mascara.png --input fotos/5_/20-40/getIT_2.tif

# 4. Exportar release científica oficial (sobrescrita desabilitada por padrão; use --studio-overwrite apenas se desejar substituir)
python -m swiftmev.cli --studio-export `
    --studio-input-dir fotos `
    --studio-output-dir dataset_releases `
    --studio-dataset-name swiftmev_fibers `
    --studio-version 0.1.0
```

---

## 6. Limitações Conhecidas

1. **Amostras Físicas Provisórias:** O campo `physical_sample_id` permanece `"unknown"` até que o caderno de laboratório de síntese seja consultado e aprovado por um pesquisador responsável.
2. **Redes Neurais:** Nenhuma arquitetura de rede neural foi treinada na Alpha 0.6B (escopo estrito da etapa). O treinamento da primeira U-Net será realizado na Alpha 0.6C.
