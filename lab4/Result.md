# Results

## Test for different KL annealing strategy

|                         PSNR                          |
| :---------------------------------------------------: |
| ![diff strategy psnr](img/diff_strategy/psnr_val.png) |

|                          Loss                           |
| :-----------------------------------------------------: |
| ![diff strategy loss](img/diff_strategy/train_loss.png) |

|                       Beta                        |                       TFR                       |
| :-----------------------------------------------: | :---------------------------------------------: |
| ![diff strategy beta](img/diff_strategy/beta.png) | ![diff strategy tfr](img/diff_strategy/tfr.png) |

### Cyclical

(gpu7-tmux0) tensorboard: Cyclical-tfr_0.0_10_0.1
Val PSNR: 26.14582061767578
Test Score: 23.83427

> Note: `w/o wandb` `old version`

![PSNR_Cyclical-tfr_0.0_10_0.1.png](img/Cyclical__tfr-0.0-10-0.1/PSNR_Cyclical-tfr_0.0_10_0.1.png)

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical --lr 0.0001 --num_epoch 200 --tfr 0 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

### Monotonic

(gpu7-tmux1) wandb: Syncing run wandering-snowflake-7

> Note: `old version`

Val PSNR: 25.5565128326416
Test Score: 23.37815

![PSNR_wandering-snowflake-7_s1lvrqi3.png](img/Monotonic__tfr-0.0-10-0.1__wandering-snowflake-7/PSNR_wandering-snowflake-7_s1lvrqi3.png)

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic --lr 0.0001 --num_epoch 200 --tfr 0 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

### None

(gpu4) wandb: Syncing run still-night-10

> Note: `old version`

Val PSNR: 34.07182693481445
Test Score: 31.61894

![PSNR_still-night-10_2do9gnvx.png](img/None__tfr-0.0-10-0.1__still-night-10/PSNR_still-night-10_2do9gnvx.png)

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without --lr 0.0001 --num_epoch 200 --tfr 0 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:1
```

(gpu4-tmux0) wandb: Syncing run wild-river-27

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without/tfr0 --lr 0.0001 --num_epoch 400 --tfr 0 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:1
```

## Test for different teacher forcing ratio

|                       Cyclical Train Loss                        |
| :--------------------------------------------------------------: |
| ![Cyclical_loss_train.png](img/diff_tfr/Cyclical_loss_train.png) |

|                        Cyclical PSNR                         |
| :----------------------------------------------------------: |
| ![Cyclical_psnr_val.png](img/diff_tfr/Cyclical_val_psnr.png) |

|                    Cyclical Beta                     |                    Cyclical TFR                    |
| :--------------------------------------------------: | :------------------------------------------------: |
| ![Cyclical_beta.png](img/diff_tfr/Cyclical_Beta.png) | ![Cyclical_tfr.png](img/diff_tfr/Cyclical_TFR.png) |

---

### tfr = 1, sde = 10, step = 0.1

**Cyclical KL Annealing**
(gpu7-tmux2) wandb: Syncing run eager-sound-11 ( Cyclical_0.5-tfr_1.0_10_0.1-20240807-033052 )
Val PSNR: 22.758983612060547

> Note: `old version`

![PSNR_eager-sound-11_g0uglg3t.png](img/Cyclical__tfr-1.0-10-0.1__eager-sound-11/PSNR_eager-sound-11_g0uglg3t.png)

<!-- TODO -->

(gpu7-tmux3) wandb: Syncing run hopeful-thunder-32

> Note: `latest version`

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical/tfr1 --lr 0.0001 --num_epoch 100 --tfr 1 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:7
```

**Monotonic KL Annealing**
(gpu7-tmux0) wandb: Syncing run hearty-armadillo-24

> Note: `latest version`

Val PSNR: 26.06894874572754

![PSNR_hearty-armadillo-24_bwy6vhj4.png](img/Monotonic__tfr-1.0-10-0.1__hearty-armadillo-24/PSNR_hearty-armadillo-24_bwy6vhj4.png)

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic/tfr1 --lr 0.0001 --num_epoch 100 --tfr 1 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

**Without KL Annealing**
(gpu4-tumx1) wandb: Syncing run peach-cosmos-23

> Note: `latest version`

Val PSNR: 20.617340087890625

![PSNR_per_frame_xo3b3zib.png](img/None__tfr-1.0-10-0.1__peach-cosmos-23/PSNR_per_frame_xo3b3zib.png)

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without/tfr1 --lr 0.0001 --num_epoch 100 --tfr 1 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:7
```

---

### tfr = 0.5, sde = 10, step = 0.05

**Cyclical KL Annealing**

(vonernue) wandb: Syncing run decent-wind-9
Val PSNR: 28.181432723999023
Test PSNR: 26.75431

> Note: `old version` `resume`

![PSNR_per_frame_uf6y1zlj.png](img/Cyclical__tfr-0.5-10-0.05__decent-wind-9/PSNR_decent-wind-9_uf6y1zlj.png)

<!-- TODO -->

(gpu7-tmux2) wandb: Syncing run soft-voice-30

> Note: `latest version`

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical/tfr05 --lr 0.0001 --num_epoch 100 --tfr 0.5 --tfr_d_step 0.05 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

**Monotonic KL Annealing**
(gpu7-tmux3) wandb: Syncing run fearless-monkey-21

> Note: `latest version`

Val PSNR: 25.91168785095215

![PSNR_fearless-monkey-21_2eth3fv6.png](img/Monotonic__tfr-0.5-10-0.05__fearless-monkey-21/PSNR_fearless-monkey-21_2eth3fv6.png)

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic/tfr05 --lr 0.0001 --num_epoch 100 --tfr 0.5 --tfr_d_step 0.05 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:4
```

**Without KL Annealing**
(gpu7-tmux2) wandb: Syncing run ancient-voice-22

> Note: `latest version`

Val PSNR: 20.92759132385254

![PSNR_ancient-voice-22_fbtsay31.png](img/None__tfr-0.5-10-0.05_ancient-voice-22/PSNR_ancient-voice-22_fbtsay31.png)

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without/tfr05 --lr 0.0001 --num_epoch 100 --tfr 0.5 --tfr_d_step 0.05 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:7
```

---

### tfr = 0

**Cyclical KL Annealing**
(dlp) wandb: Syncing run glorious-sun-6

> Note: `old version`

Val PSNR: 22.769935607910156

![PSNR_glorious-sun-6_u2vg0sx9.png](img/Cyclical__tfr-0.0-10-0.1__glorious-sun-6/PSNR_glorious-sun-6_u2vg0sx9.png)

<!-- TODO -->

(gpu4-tmux1) wandb: Syncing run still-armadillo-31

> Note: `latest version`

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Cyclical/tfr0 --lr 0.0001 --num_epoch 100 --tfr 0 --kl_anneal_type Cyclical --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

**Monotonic KL Annealing**

<!-- TODO -->

(gpu7-tmux0) wandb: Syncing run sunny-snow-28

> Note: `latest version`

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Monotonic/tfr0 --lr 0.0001 --num_epoch 100 --tfr 0 --kl_anneal_type Monotonic --kl_anneal_ratio 0.5 --wandb --device cuda:3
```

**Without KL Annealing**

<!-- TODO -->

(gpu4-tmux1) wandb: Syncing run dainty-bee-29

> Note: `latest version`

```bash
python Trainer.py --DR ../dataset --save_root ../saved_models/Without/tfr0 --lr 0.0001 --num_epoch 100 --tfr 0 --kl_anneal_type None --kl_anneal_ratio 0.5 --wandb --device cuda:4
```
