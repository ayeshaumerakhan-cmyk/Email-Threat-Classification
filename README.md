# Email Threat Classification

A reproducible capstone prototype that classifies email as **spam** or **ham** (not spam) from observable message characteristics. It downloads the public SpamAssassin corpus, extracts a transparent feature table, trains and evaluates a Logistic Regression baseline, and provides a Streamlit prediction demo.

> This is an educational prototype, not a production security control. Its corpus is historical, and a prediction must not be treated as proof that a message is safe or malicious.

## Quick start

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.email_threat.dataset
python -m src.email_threat.train
streamlit run app.py
```

On first run, the dataset command downloads the three public SpamAssassin corpus archives into `data/raw/`, then writes the derived feature table to `data/processed/email_features.csv`. The archives are kept locally and are not committed to this project. Re-run the command to rebuild the feature table from the downloaded archives.

To train with another test-set fraction or seed:

```powershell
python -m src.email_threat.train --test-size 0.2 --seed 42
```

Training writes the fitted model to `models/email_threat_model.joblib` and evaluation outputs to `reports/`. Run the Streamlit app after training. The app expects the trained model artifact; if it is missing, it shows the training command.

Run the automated checks with:

```powershell
python -m pytest -q
```

## Project structure

```text
app.py                         Streamlit email-analysis demo
src/email_threat/features.py   Shared training/inference feature extraction
src/email_threat/dataset.py    Download, parse, and label public email corpus
src/email_threat/train.py      Stratified split, model training, and evaluation
tests/                         Feature extraction and dataset tests
data/raw/                      Downloaded source archives (not committed)
data/processed/                Reproducible engineered dataset
models/                        Trained model artifact
reports/                       Metrics, plots, and technical paper
```

## Dataset and provenance

Source: [Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/), specifically the `20030228_spam`, `20030228_easy_ham`, and `20030228_hard_ham` archives. The corpus contains raw messages collected for spam-filtering research. Archive folders provide the labels: spam is `1`; easy ham and hard ham are `0`. The dataset builder records the source archive name for each row and skips non-message entries.

The project engineers 15 numeric features from the email subject, body, sender, and MIME structure. It does **not** use external APIs or send email contents anywhere. Dataset messages are old and may include personal information; the raw archives should be handled as research data and not republished. Cite the SpamAssassin public corpus when presenting results.

## Method

The baseline is a class-weighted Logistic Regression inside a StandardScaler pipeline. A stratified 80/20 holdout preserves the class proportions. The script reports accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix, and saves ROC, confusion-matrix, and coefficient plots. The fixed seed makes the split repeatable. The test set is used for final reporting, not model selection.

The Streamlit form collects the same inputs used by the shared feature extractor. Training and inference therefore use the same feature names and preprocessing logic.

## Limitations and responsible use

- The corpus dates from 2002–2003 and does not represent current phishing tactics, languages, or mail clients.
- Spam/ham labels are not the same as a verified malicious/safe verdict; false negatives and false positives are possible.
- Hand-engineered indicators are intentionally interpretable but are not a substitute for URL reputation, attachment sandboxing, authentication checks, or human review.
- Do not submit confidential or personal messages to a demo unless you control the environment and have permission.
- The model is a classroom prototype and must not be used to block, quarantine, or automatically trust real email.

## Reproducibility and deliverables

- Dataset citation and source archives: this README and the dataset builder.
- Engineered data: `data/processed/email_features.csv`.
- Trained model: `models/email_threat_model.joblib`.
- Metrics and figures: `reports/`.
- Technical paper: `reports/technical_paper.md`.

## Baseline run (seed 42)

The current stratified holdout contains 651 records (551 ham, 100 spam). Logistic Regression achieved 78.2% accuracy, 38.0% spam precision, 66.0% spam recall, 48.2% spam F1, and 0.777 ROC-AUC. Its confusion matrix (actual rows: ham, spam; predicted columns: ham, spam) is `[[443, 108], [34, 66]]`. The low spam precision means many legitimate messages are flagged; the model is included as a transparent baseline, not presented as deployment-ready. See the technical paper and `reports/metrics.json` for the full evaluation.

## License and attribution

The project code is provided for educational use. The SpamAssassin corpus is a separate third-party dataset; consult the [source project](https://spamassassin.apache.org/old/publiccorpus/) for its terms and attribution requirements. Do not assume this code license applies to the corpus.
