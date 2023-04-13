#!/bin/bash

conda activate robustness

mkdir "$PWD/explanations_full/"
mkdir "$PWD/explanations_limited/"

python3 src/explanations/compare_captum.py checkpoint_path=/models/ResNet50_full.pth n_aug=40 n_to_expl=500 save_expl=True path_expl="$PWD/explanations_full/"
python3 src/explanations/compare_captum.py checkpoint_path=/models/ResNet50_limited.pth n_aug=40 n_to_expl=500 save_expl=True path_expl="$PWD/explanations_limited/"

python3 src/explanations/process_results.py 'models_to_process=[ResNet50_full, ResNet50_limited]' probab_drop=0.1 only_correct=True

# Exaluate pixel flipping
python3 src/explanations/pixel_flipping.py Gradients InputGradients IntegratedGradients GuidedBackprop Deconvolution zennit-EpsilonPlusFlat zennit-EpsilonGammaBox zennit-EpsilonAlpha2Beta1Flat "$PWD/explanations_full/" ./reports/figures ./outputs/ResNet50_new_all_ImageNet_AddToBrightness_Gradients.csv --aug AddToBrightness AddToHue AddToSaturation Rotate Translate Scale
python3 src/explanations/pixel_flipping.py Gradients InputGradients IntegratedGradients GuidedBackprop Deconvolution zennit-EpsilonPlusFlat zennit-EpsilonGammaBox zennit-EpsilonAlpha2Beta1Flat "$PWD/explanations_limited/" ./reports/figures ./outputs/ResNet50_new_all_ImageNet_AddToBrightness_Gradients.csv --aug AddToBrightness AddToHue AddToSaturation Rotate Translate Scale

# Plot explanations
mkdir "$PWD/explanations__figures_full/"
mkdir "$PWD/explanations_figures_limited/"
python3 src/visualization/plot_explanations.py "$PWD/explanations_full/" ILSVRC2012_val_00000094 /"$PWD/explanations_figures_full/" --explainer Gradients InputGradients IntegratedGradients GuidedBackprop Deconvolution zennit-EpsilonPlusFlat zennit-EpsilonGammaBox zennit-EpsilonAlpha2Beta1Flat --aug AddToBrightness AddToHue AddToSaturation Rotate Translate Scale
python3 src/visualization/plot_explanations.py "$PWD/explanations_full/" ILSVRC2012_val_00000094 /"$PWD/explanations_figures_limited/" --explainer Gradients InputGradients IntegratedGradients GuidedBackprop Deconvolution zennit-EpsilonPlusFlat zennit-EpsilonGammaBox zennit-EpsilonAlpha2Beta1Flat --aug AddToBrightness AddToHue AddToSaturation Rotate Translate Scale
