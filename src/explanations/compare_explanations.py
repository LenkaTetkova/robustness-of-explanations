import numpy as np
import quantus

from src.explanations.pixel_flipping import PixelFlipping


def pixel_flipping_AOC(images, labels, explanations, model, features_in_step,
                       to_plot=False, perturb_baseline="mean"):
    pixel_flipping = PixelFlipping(**{
            "features_in_step": features_in_step,
            "perturb_baseline": perturb_baseline,
            "perturb_func": quantus.functions.perturb_func.baseline_replacement_by_indices,
            "disable_warnings": True,
        })
    scores = pixel_flipping(model=model,
                            x_batch=images,
                            y_batch=labels,
                            a_batch=explanations,
                            **{"device": "cpu"})
    if to_plot:
        pixel_flipping.plot(None, y_batch=labels.flatten(), scores=scores)
    chosen_scores = [scores[i][:len(scores[0])//5] for i in range(len(scores))]
    PF_score = [1-np.trapz(sc, dx=1/(len(sc)-1)) for sc in chosen_scores]
    PF_score = np.mean(PF_score)
    return PF_score, scores
