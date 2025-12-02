models=("clip" "plip" "quilt" "biomedclip" "mi_zero_pubmedbert" "mi_zero_bioclinicalbert" \
        "conch" "keep" "pathclip")

datasets=(
          "BACH" "BACH_detection" "BCNB_metastasis" "BCNB_tumor" "BRACS" \
          "BreakHist" "CAMEL" "Chaoyang_detection" \
          "Chaoyang_phenotyping" "CRC_ICM_grade" "CRC_ICM_stage" "CRC_ICM_subtype" "CRC_TP" \
          "DataBiox" "DigestPath" \
          "GasHisSDB" "GlaS_binary" "GlaS_grading" \
          "Kather" "Kather5K" "LC25000_colon" "LC25000_lung" "LC25000_lung_detection" "MHIST" \
          "Osteosarcoma" "Osteosarcoma_detection" "PanNuke_subtyping" "PanNuke" "PatchGastric_subtypes" \
          "PatchGastric" "PCam" "Prostate_Harvard_detection" "Prostate_Harvard_gleason" \
          "Prostate_Harvard_isup" "RenalCell_Lymph" "RenalCell_Tissue" "SICAP_detection" "SICAP_grading" \
          "SkinCancer_subtyping" "SkinCancer" "TCGA_TIL" "WSSS4LUAD" \
          "MSI_MSS_binary" "MSI_MSS_cancerType" "MSI_MSS_CRC" "MSI_MSS_FlashFrozen" "MSI_MSS_STAD")


text_errors=("None") # "remove" "replace" "swap"


# Iterating over available models
# Reproducing zero-shot experiments

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for text_error in "${text_errors[@]}"; do
      python3 scripts/zero_shot_evaluation.py --model_name=$model --dataset=$dataset --text_error=$text_error --adversarial=False
    done
  done
done
