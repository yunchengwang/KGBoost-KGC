# Best setting for FB15k-237
python KGBoost.py --gpu 0 \
                  --dataset FB15k-237 \
                  --pretrained_emb RotatE \
                  --max_depth 5 \
                  --negative_size 64 \
                  --n_estimators 1000 \
                  --sampling_1 rcwc \
                  --sampling_2 none \
                  --lcwa \
                  --lcw_threshold 0.0
python KGBoost.py --gpu 0 \
                  --dataset FB15k-237 \
                  --pretrained_emb TransE \
                  --max_depth 5 \
                  --negative_size 64 \
                  --n_estimators 1000 \
                  --sampling_1 rcwc \
                  --sampling_2 none \
                  --lcwa \
                  --lcw_threshold 0.0

# Best setting for WN18RR
python KGBoost.py --gpu 0 \
                  --dataset WN18RR \
                  --pretrained_emb RotatE \
                  --max_depth 3 \
                  --negative_size 32 \
                  --n_estimators 1500 \
                  --sampling_1 naive \
                  --sampling_2 adv \
                  --augment \
                  --lcwa \
                  --lcw_threshold 0.8
python KGBoost.py --gpu 0 \
                  --dataset WN18RR \
                  --pretrained_emb TransE \
                  --max_depth 3 \
                  --negative_size 32 \
                  --n_estimators 1500 \
                  --sampling_1 rcwc \
                  --sampling_2 none \
                  --augment \
                  --lcwa \
                  --lcw_threshold 0.8

# Best setting for FB15K
python KGBoost.py --gpu 0 \
                  --dataset FB15K \
                  --pretrained_emb RotatE \
                  --max_depth 5 \
                  --negative_size 32 \
                  --n_estimators 1000 \
                  --sampling_1 rcwc \
                  --sampling_2 none \
                  --augment \
                  --lcwa \
                  --lcw_threshold 0.0
python KGBoost.py --gpu 0 \
                  --dataset FB15K \
                  --pretrained_emb TransE \
                  --max_depth 5 \
                  --negative_size 32 \
                  --n_estimators 1000 \
                  --sampling_1 rcwc \
                  --sampling_2 none \
                  --augment \
                  --lcwa \
                  --lcw_threshold 0.0

# Best setting for WN18
python KGBoost.py --gpu 0 \
                  --dataset WN18 \
                  --pretrained_emb RotatE \
                  --max_depth 3 \
                  --negative_size 32 \
                  --n_estimators 1500 \
                  --sampling_1 naive \
                  --sampling_2 adv \
                  --augment \
                  --lcwa \
                  --lcw_threshold 0.8
python KGBoost.py --gpu 0 \
                  --dataset WN18 \
                  --pretrained_emb TransE \
                  --max_depth 10 \
                  --negative_size 32 \
                  --n_estimators 1000 \
                  --sampling_1 naive \
                  --sampling_2 adv \
                  --augment \
                  --lcwa \
                  --lcw_threshold 0.8