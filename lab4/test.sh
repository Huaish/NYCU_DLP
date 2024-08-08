# Best Test Command
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr0_still-night-10_ep200/final-still-night-10.ckpt --device cuda:7



# ----------------- #

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr0_ep200/final-Cyclical.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr0_ep200/final-Cyclical.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Monotonic/tfr0_wandering-snowflake-7_ep200/final-wandering-snowflake-7.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Monotonic/tfr0_wandering-snowflake-7_ep200/final-wandering-snowflake-7.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Without/tfr0_still-night-10_ep200/final-still-night-10.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr0_still-night-10_ep200/final-still-night-10.ckpt --device cuda:7

# ----------------- #

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr0_glorious-sun-6_ep100/final-glorious-sun-6.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr0_glorious-sun-6_ep100/final-glorious-sun-6.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr05_decent-wind-9_ep100/final-decent-wind-9.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr05_decent-wind-9_ep100/final-decent-wind-9.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr1_eager-sound-11_ep100/final-eager-sound-11.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr1_eager-sound-11_ep100/final-eager-sound-11.ckpt --device cuda:3

# ----------------- #

# wait for training to finish

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Monotonic/tfr05_fearless-monkey-21_ep100/final-fearless-monkey-21.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Monotonic/tfr05_decent-wind-9_ep100/final-fearless-monkey-21.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Monotonic/tfr1_hearty-armadillo-24_ep100/final-hearty-armadillo-24.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Monotonic/tfr1_hearty-armadillo-24_ep100/final-hearty-armadillo-24.ckpt --device cuda:3

# ----------------- #

# wait for training to finish

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Without/tfr05_ancient-voice-22_ep100/final-ancient-voice-22.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr05_ancient-voice-22_ep100/final-ancient-voice-22.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Without/tfr1-peach-cosmos-23_ep100/final-peach-cosmos-23.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr1-peach-cosmos-23_ep100/final-peach-cosmos-23.ckpt --device cuda:3