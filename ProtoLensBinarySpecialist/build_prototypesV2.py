"""
build_prototypes.py (v2 - clustering per categoria)

Costruisce i prototipi di ProtoLens in modo SUPERVISIONATO rispetto alle 8
categorie originali di LexGLUE unfair_tos, invece che con un unico K-Means
"cieco" su tutto il dataset.

Per ciascuna categoria c (0..7):
    1. si prendono SOLO le clausole (righe) in cui quella categoria e' presente
       (nessuno split di frase: ogni riga di *_multilabel.csv e' gia' una
       clausola/unita' testuale completa)
    2. si calcolano gli embedding con lo stesso SentenceTransformer del modello
    3. si esegue K-Means con k = prototypes_per_class SOLO su quelle clausole
    4. per ciascun cluster si salva il centroide + un pool di clausole vicine

Il numero totale di prototipi non e' piu' un parametro arbitrario:
    prototype_num = num_categories * prototypes_per_class

Output (stesso formato/path atteso da PLens.py, piu' un file nuovo):
    <dataset>_cluster_<K>_centers.npy         (shape: K x hidden_dim)
    <dataset>_cluster_<K>_to_sub_sentence.csv (K righe, pool di clausole per prototipo)
    <dataset>_prototype_metadata.json         (NUOVO: prototipo -> categoria/cluster/frase rappresentativa)

Il prototipo con indice i corrisponde a:
    categoria    = i // prototypes_per_class
    cluster_id   = i %  prototypes_per_class
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

#NUM_CATEGORIES = 8  # categorie 0..7 di unfair_tos


# def build_prototypes_per_category(
#     multilabel_csv_path,
#     base_folder,
#     data_set,
#     bert_model_name,
#     prototypes_per_class=3,
#     candidates_per_prototype=15,
#     seed=42,
# ):
#     from sentence_transformers import SentenceTransformer

#     print(f"[1/4] Carico le clausole etichettate da: {multilabel_csv_path}")
#     df = pd.read_csv(multilabel_csv_path)
#     label_cols = [f"label_{c}" for c in range(NUM_CATEGORIES)]
#     missing = [c for c in label_cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"Colonne mancanti nel file multilabel: {missing}")

#     print(f"[2/4] Carico il modello di embedding: {bert_model_name}")
    
#     model = SentenceTransformer(f'sentence-transformers/{bert_model_name}')

#     #model = SentenceTransformer('/content/all-mpnet-base-v2')

#     prototype_num = NUM_CATEGORIES * prototypes_per_class
#     all_centers = []
#     all_pool_rows = []
#     metadata = {}

#     for category in range(NUM_CATEGORIES):
#         col = f"label_{category}"
#         cat_texts = df.loc[df[col] == 1, 'review'].tolist()
#         print(f"[3/4] Categoria {category}: {len(cat_texts)} clausole disponibili")

#         if len(cat_texts) < prototypes_per_class:
#             raise ValueError(
#                 f"La categoria {category} ha solo {len(cat_texts)} clausole, "
#                 f"insufficienti per {prototypes_per_class} prototipi. "
#                 f"Riduci prototypes_per_class oppure escludi questa categoria."
#             )

#         cat_embeddings = model.encode(
#             cat_texts, normalize_embeddings=True,
#             convert_to_numpy=True, show_progress_bar=False, batch_size=64
#         )

#         k = prototypes_per_class
#         kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
#         labels = kmeans.fit_predict(cat_embeddings)
#         centers = kmeans.cluster_centers_.astype(np.float32)

#         for cluster_id in range(k):
#             prototype_id = category * prototypes_per_class + cluster_id
#             idx_in_cluster = np.where(labels == cluster_id)[0]

#             if len(idx_in_cluster) == 0:
#                 sims = cosine_similarity(centers[cluster_id:cluster_id + 1], cat_embeddings)[0]
#                 top_idx = np.argsort(sims)[-candidates_per_prototype:][::-1]
#             else:
#                 cluster_embeddings = cat_embeddings[idx_in_cluster]
#                 sims = cosine_similarity(centers[cluster_id:cluster_id + 1], cluster_embeddings)[0]
#                 order = np.argsort(sims)[::-1]
#                 top_idx = idx_in_cluster[order[:candidates_per_prototype]]

#             selected = [cat_texts[i] for i in top_idx]
#             while len(selected) < candidates_per_prototype:
#                 selected.append(selected[-1] if selected else "")
#             selected = selected[:candidates_per_prototype]

#             all_centers.append(centers[cluster_id])
#             all_pool_rows.append(selected)
#             metadata[str(prototype_id)] = {
#                 "category": category,
#                 "cluster": cluster_id,
#                 "sentence": selected[0],  # clausola piu' vicina al centroide
#                 "pool_size": len(idx_in_cluster),
#             }

#     centers_arr = np.stack(all_centers, axis=0)  # (prototype_num, hidden_dim)

#     out_dir = os.path.join(base_folder, data_set, bert_model_name)
#     os.makedirs(out_dir, exist_ok=True)

#     centers_path = os.path.join(out_dir, f"{data_set}_cluster_{prototype_num}_centers.npy")
#     csv_path = os.path.join(out_dir, f"{data_set}_cluster_{prototype_num}_to_sub_sentence.csv")
#     metadata_path = os.path.join(out_dir, f"{data_set}_prototype_metadata.json")

#     np.save(centers_path, centers_arr)
#     pd.DataFrame(all_pool_rows).to_csv(csv_path, header=False)
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=2, ensure_ascii=False)

#     print(f"[4/4] Fatto. prototype_num = {prototype_num} "
#           f"({NUM_CATEGORIES} categorie x {prototypes_per_class} prototipi ciascuna)")
#     print(f"  centers   -> {centers_path}  (shape {centers_arr.shape})")
#     print(f"  sentences -> {csv_path}")
#     print(f"  metadata  -> {metadata_path}")

#     return centers_path, csv_path, metadata_path, prototype_num

NUM_CATEGORIES = 8  # categorie 0..7 di unfair_tos
FAIR_CATEGORY = 8   # nona categoria (fair)


def build_prototypes_per_category(
    multilabel_csv_path,
    base_folder,
    data_set,
    bert_model_name,
    prototypes_per_class=3,
    prototypes_for_fair=15,  # NUOVO: Prototipi dedicati solo alla classe fair
    candidates_per_prototype=15,
    seed=42,
):
    from sentence_transformers import SentenceTransformer

    print(f"[1/4] Carico le clausole etichettate da: {multilabel_csv_path}")
    df = pd.read_csv(multilabel_csv_path)
    label_cols = [f"label_{c}" for c in range(NUM_CATEGORIES)]
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel file multilabel: {missing}")

    print(f"[2/4] Carico il modello di embedding: {bert_model_name}")
    model = SentenceTransformer(f'sentence-transformers/{bert_model_name}')

    total_categories = NUM_CATEGORIES + 1  # 8 categorie unfair + 1 categoria fair
    
    # NUOVO: Calcolo dinamico del totale dei prototipi
    # Esempio: (8 * 3) + 15 = 39 prototipi totali
    prototype_num = (NUM_CATEGORIES * prototypes_per_class) + prototypes_for_fair
    
    all_centers = []
    all_pool_rows = []
    metadata = {}

    for category in range(total_categories):
        # NUOVO: Settiamo il numero di cluster (k) e l'ID di base in base alla categoria
        if category < NUM_CATEGORIES:
            # Categorie Unfair 0..7
            col = f"label_{category}"
            cat_texts = df.loc[df[col] == 1, 'review'].tolist()
            k = prototypes_per_class
            base_prototype_id = category * prototypes_per_class
        else:
            # Categoria 8: Fair (tutte le etichette unfair sono pari a 0)
            is_fair = (df[label_cols].sum(axis=1) == 0)
            cat_texts = df.loc[is_fair, 'review'].tolist()
            k = prototypes_for_fair
            base_prototype_id = NUM_CATEGORIES * prototypes_per_class  # Parte da 24

        print(f"[3/4] Categoria {category} ({'Unfair ' + str(category) if category < 8 else 'FAIR'}): {len(cat_texts)} clausole disponibili (K={k})")

        if len(cat_texts) < k:
            raise ValueError(
                f"La categoria {category} ha solo {len(cat_texts)} clausole, "
                f"insufficienti per {k} prototipi. Riduci k."
            )

        cat_embeddings = model.encode(
            cat_texts, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, batch_size=64
        )

        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(cat_embeddings)
        centers = kmeans.cluster_centers_.astype(np.float32)

        for cluster_id in range(k):
            # NUOVO: L'ID del prototipo si appoggia all'indice di base corretto
            prototype_id = base_prototype_id + cluster_id
            idx_in_cluster = np.where(labels == cluster_id)[0]

            if len(idx_in_cluster) == 0:
                sims = cosine_similarity(centers[cluster_id:cluster_id + 1], cat_embeddings)[0]
                top_idx = np.argsort(sims)[-candidates_per_prototype:][::-1]
            else:
                cluster_embeddings = cat_embeddings[idx_in_cluster]
                sims = cosine_similarity(centers[cluster_id:cluster_id + 1], cluster_embeddings)[0]
                order = np.argsort(sims)[::-1]
                top_idx = idx_in_cluster[order[:candidates_per_prototype]]

            selected = [cat_texts[i] for i in top_idx]
            while len(selected) < candidates_per_prototype:
                selected.append(selected[-1] if selected else "")
            selected = selected[:candidates_per_prototype]

            all_centers.append(centers[cluster_id])
            all_pool_rows.append(selected)
            metadata[str(prototype_id)] = {
                "category": category,
                "cluster": cluster_id,
                "sentence": selected[0],  # clausola più vicina al centroide
                "pool_size": len(idx_in_cluster),
            }

    centers_arr = np.stack(all_centers, axis=0)  # (prototype_num, hidden_dim)

    out_dir = os.path.join(base_folder, data_set, bert_model_name)
    os.makedirs(out_dir, exist_ok=True)

    centers_path = os.path.join(out_dir, f"{data_set}_cluster_{prototype_num}_centers.npy")
    csv_path = os.path.join(out_dir, f"{data_set}_cluster_{prototype_num}_to_sub_sentence.csv")
    metadata_path = os.path.join(out_dir, f"{data_set}_prototype_metadata.json")

    np.save(centers_path, centers_arr)
    pd.DataFrame(all_pool_rows).to_csv(csv_path, header=False)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[4/4] Fatto. prototype_num = {prototype_num}")
    print(f"  centers   -> {centers_path}  (shape {centers_arr.shape})")
    print(f"  sentences -> {csv_path}")
    print(f"  metadata  -> {metadata_path}")

    return centers_path, csv_path, metadata_path, prototype_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multilabel_csv', type=str, required=True,
                         help="Path al file *_multilabel.csv")
    parser.add_argument('--base_folder', type=str, required=True)
    parser.add_argument('--data_set', type=str, required=True)
    parser.add_argument('--bert_model_name', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--prototypes_per_class', type=int, default=3)
    parser.add_argument('--prototypes_for_fair', type=int, default=15) # NUOVO
    parser.add_argument('--candidates_per_prototype', type=int, default=15)
    cli_args = parser.parse_args()

    build_prototypes_per_category(
        multilabel_csv_path=cli_args.multilabel_csv,
        base_folder=cli_args.base_folder,
        data_set=cli_args.data_set,
        bert_model_name=cli_args.bert_model_name,
        prototypes_per_class=cli_args.prototypes_per_class,
        prototypes_for_fair=cli_args.prototypes_for_fair,
        candidates_per_prototype=cli_args.candidates_per_prototype,
    )
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--multilabel_csv', type=str, required=True,
#                          help="Path al file *_multilabel.csv generato da build_or_dataset.py "
#                               "(tipicamente train_multilabel.csv, coerente col train set)")
#     parser.add_argument('--base_folder', type=str, required=True)
#     parser.add_argument('--data_set', type=str, required=True)
#     parser.add_argument('--bert_model_name', type=str, default='all-mpnet-base-v2')
#     parser.add_argument('--prototypes_per_class', type=int, default=3)
#     parser.add_argument('--candidates_per_prototype', type=int, default=15)
#     cli_args = parser.parse_args()

#     build_prototypes_per_category(
#         multilabel_csv_path=cli_args.multilabel_csv,
#         base_folder=cli_args.base_folder,
#         data_set=cli_args.data_set,
#         bert_model_name=cli_args.bert_model_name,
#         prototypes_per_class=cli_args.prototypes_per_class,
#         candidates_per_prototype=cli_args.candidates_per_prototype,
#     )