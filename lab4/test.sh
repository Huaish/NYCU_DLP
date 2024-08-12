# Best Test Command
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr0_wild-river-27_ep400/final_model.ckpt



# ----------------- #

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr0_ep200/final-Cyclical.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr0_ep200/final-Cyclical.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Monotonic/tfr0_wandering-snowflake-7_ep200/final-wandering-snowflake-7.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Monotonic/tfr0_wandering-snowflake-7_ep200/final-wandering-snowflake-7.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Without/tfr0_still-night-10_ep200/final-still-night-10.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr0_still-night-10_ep200/final-still-night-10.ckpt --device cuda:7

# ----------------- #

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr0_still-armadillo-31_ep100/best-still-armadillo-31.ckpt
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr0_still-armadillo-31_ep100/best-still-armadillo-31.ckpt

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr05_warm-resonance-35_ep100/best-warm-resonance-35.ckpt
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr05_warm-resonance-35_ep100/best-warm-resonance-35.ckpt 

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Cyclical/tfr1_playful-yogurt-36_ep100/best-playful-yogurt-36.ckpt
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Cyclical/tfr1_playful-yogurt-36_ep100/best-playful-yogurt-36.ckpt

# ----------------- #

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Monotonic/tfr0_northern-sky-37_ep100/best-northern-sky-37.ckpt
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Monotonic/tfr0_northern-sky-37_ep100/best-northern-sky-37.ckpt

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Monotonic/tfr05_fearless-monkey-21_ep100/final-fearless-monkey-21.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Monotonic/tfr05_decent-wind-9_ep100/final-fearless-monkey-21.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Monotonic/tfr1_hearty-armadillo-24_ep100/final-hearty-armadillo-24.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Monotonic/tfr1_hearty-armadillo-24_ep100/final-hearty-armadillo-24.ckpt --device cuda:3

# ----------------- #

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Without/tfr0_dainty-bee-29/best_model.ckpt
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr0_dainty-bee-29/best_model.ckpt

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Without/tfr05_ancient-voice-22_ep100/final-ancient-voice-22.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr05_ancient-voice-22_ep100/final-ancient-voice-22.ckpt --device cuda:3

python Trainer.py --DR ../dataset --save_root ../submission/ --test --ckpt_path ../saved_models/Without/tfr1-peach-cosmos-23_ep100/final-peach-cosmos-23.ckpt --device cuda:3
python Tester.py --DR ../dataset --save_root ../submission/ --ckpt_path ../saved_models/Without/tfr1-peach-cosmos-23_ep100/final-peach-cosmos-23.ckpt --device cuda:3