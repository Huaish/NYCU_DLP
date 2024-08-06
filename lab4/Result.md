# Resutls

## Test for different KL annealing strategy

### Cyclical 

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical --lr 0.0001 --num_epoch 200 --tfr 0 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

### Monotonic

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic --lr 0.0001 --num_epoch 200 --tfr 0 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:1
```

### None

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without --lr 0.0001 --num_epoch 200 --tfr 0 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

## Test for different teacher forcing ratio

### tfr = 1, sde = 10, step = 0.1

**Cyclical KL Annealing**

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical --lr 0.0001 --num_epoch 100 --tfr 1 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

**Monotonic KL Annealing**

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic --lr 0.0001 --num_epoch 100 --tfr 1 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:1
```

**Without KL Annealing**

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without --lr 0.0001 --num_epoch 100 --tfr 1 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

---

### tfr = 0.5, sde = 10, step = 0.05

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical --lr 0.0001 --num_epoch 100 --tfr 0.5 --tfr_d_step 0.05 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

**Monotonic KL Annealing**

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic --lr 0.0001 --num_epoch 100 --tfr 0.5 --tfr_d_step 0.05 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:1
```

**Without KL Annealing**

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without --lr 0.0001 --num_epoch 100 --tfr 0.5 --tfr_d_step 0.05 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

---

### tfr = 0

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical --lr 0.0001 --num_epoch 100 --tfr 0 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

**Monotonic KL Annealing**

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic --lr 0.0001 --num_epoch 100 --tfr 0 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:1
```

**Without KL Annealing**

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without --lr 0.0001 --num_epoch 100 --tfr 0 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

