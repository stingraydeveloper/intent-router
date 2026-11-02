# intent-router

Sorts incoming customer messages into 150 intents, and hands a message to a person when the model is not sure. The question it answers: if a classifier is going to act without a human, which model makes the fewest wrong moves?

Data is [CLINC150](https://github.com/clinc/oos-eval) (CC BY 3.0): 15,000 labelled training messages, and a test set of 4,500 in-scope and 1,000 out-of-scope messages. The out-of-scope ones matter most, because a real inbox is full of things nobody trained for.

## Results

From `results/metrics.json`, on the held-out test set. The confidence threshold for each model was picked on the validation set, aiming for 97% precision.

| Model | In-scope accuracy | Routed automatically | Out-of-scope caught | Wrong automatic moves |
|---|---|---|---|---|
| TF-IDF + logistic regression | 92.5% | 87.9% | 89.9% | 3.7% |
| Hand-written neural network (NumPy) | 88.4% | 76.0% | 88.3% | 3.9% |
| MiniLM embeddings + logistic regression | 95.3% | 96.2% | 73.0% | 7.1% |

Three things I took from this:

- **Accuracy picked the wrong model.** Embeddings score highest, then let 27% of out-of-scope messages through with confidence. TF-IDF made about half as many wrong automatic moves.
- **The threshold missed its target.** All three aimed for 97% and landed between 92% and 95% on test (see `routed_precision`). Validation has far fewer out-of-scope messages than test, so a threshold tuned there is too generous. Tune against the mix your real inbox has.
- **Simple won on speed as well.** TF-IDF takes 0.04 ms per message on a laptop. A hosted LLM call takes hundreds of milliseconds and costs money per message, so a classifier is worth trying first when the labels are stable.

## The hand-written network

`intent_router/scratch_nn.py` is a one-hidden-layer softmax classifier with backpropagation and Adam in plain NumPy. `tests/test_scratch_nn.py` checks its gradients against a numerical estimate and checks it can learn XOR. It loses to logistic regression here, which is the honest result on 400 compressed features.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/pytest -q
PYTHONPATH=. .venv/bin/python train.py            # writes results/metrics.json
PYTHONPATH=. .venv/bin/python export_browser.py   # writes the model used by the web demo
```

`export_browser.py` prunes a word-only model to 140 weights per intent so it can run in a web page. Its own numbers are in `results/browser_model.json`, and `tests/browser_parity.test.mjs` checks the JavaScript version against Python on 400 messages. You can [try it in your browser](https://stingraydeveloper.github.io/#router).

Built with Claude Code.
