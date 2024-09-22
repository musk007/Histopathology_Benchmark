from reproducibility.metrics import retrieval_metrics
import logging
import os
import json

class ImageRetrieval:

    def __init__(self):
        pass

    def retrieval(self, image_embeddings, text_embeddings, model_name, ds_name):

        best_scores = []

        for t in text_embeddings:
            arr = t.dot(image_embeddings.T)

            best = arr.argsort()[-50:][::-1]

            best_scores.append(best)

        targets = list(range(0, len(image_embeddings)))

        test_metrics = retrieval_metrics(targets, best_scores)
        train_metrics = retrieval_metrics(targets, best_scores)


        test_metrics["split"] = "test"
        folder_path = f"/home/roba.majzoub/research/new_plip/plip/results/{ds_name}"
        if os.path.isdir(folder_path) == False:
            os.mkdir(folder_path)
        f= open(os.path.join(folder_path,f"{model_name}_retrieval.json"), 'w')
        json.dump(test_metrics, f)
        f.close()
        train_metrics["split"] = "train"

        logging.info(f"Retrieval Done")

        return train_metrics, test_metrics
