# Can Sequence Models Think Deeper Than They Were Taught?

ELEC4542 project. Topic 5 (ListOps-64) plus our own extensions. Everything runs on a **free Colab T4** (~2-3 h).

## What's in it
| Part | Question | Script |
|---|---|---|
| Base comparison (brief) | RNN vs LSTM vs Transformer vs Linear Attention: accuracy, learning curves, cost | `train.py`, `analyze.py` |
| Ext. 1: depth generalisation | Trained only on shallow expressions, can they solve deeper ones? | `train.py --train_max_depth D` |
| Ext. 2: probing | Is each sub-answer (e.g. `[SM 7 6 ]` = 3) readable from the hidden state, even though only the final answer was trained? Controls: untrained models + majority baseline | `probe.py` |
| Ext. 3: attention | Does the Transformer look inside its own brackets? (bracket-focus score vs uniform) | `attention.py` |
| Ext. 4: **Looped Transformer (ours)** | One shared layer applied K times. Change K at test time: does thinking longer help deeper expressions? | `models.py` (`looped`), `loops.py` |

The models only ever see the raw token sequence. The reference evaluator and sub-answers are used **only** for label checks and probing, never as model inputs (as the brief requires).

## Run
Open `run_colab.ipynb` in Colab with a T4 GPU, upload all `.py` files, and run top to bottom. Or:

    pip install torch datasets scikit-learn matplotlib numpy
    python data.py
    for m in rnn lstm transformer linear looped; do python train.py --model $m --run_name base_$m; done
    for m in rnn lstm transformer linear looped; do python train.py --model $m --run_name depth3_$m --train_max_depth 3; done
    python probe.py; python attention.py
    python loops.py --run base_looped; python loops.py --run depth3_looped
    python analyze.py

Run names matter: base_<model>, depth<D>_<model>, pos_<kind>_transformer (optional PE ablation).
Offline smoke test without downloading data: `LISTOPS_FAKE=1 python train.py --model looped --run_name t --max_steps 50 --no_test`.
