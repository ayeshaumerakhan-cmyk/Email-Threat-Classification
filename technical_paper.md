# Email Threat Classification: An Interpretable Spam/Ham Baseline

**Capstone technical paper**  
**Domain:** Cybersecurity · **Task:** Binary classification · **Dataset:** Apache SpamAssassin Public Corpus

## Abstract

This project builds a reproducible, end-to-end baseline for classifying email as spam or ham (not spam). The SpamAssassin public corpus supplies labeled historical raw email. A deterministic parser and shared feature extractor derive 15 numeric features from message length, URLs, MIME attachments, subject/body keyword indicators, sender traits, HTML presence, and capitalization. A class-weighted Logistic Regression pipeline is trained on a stratified 80/20 split and evaluated with accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix. On the fixed seed-42 holdout of 651 emails, the model reached 78.2% accuracy, 38.0% precision, 66.0% recall, 48.2% F1, and 0.777 ROC-AUC. The results demonstrate a working, interpretable pipeline but also show that this small, historical, hand-engineered feature set is not suitable for operational email filtering.

## 1. Problem definition and objective

Spam and phishing messages can exploit users through deceptive content and links. This capstone asks whether simple, observable email characteristics can help distinguish unwanted messages from legitimate ones. The implemented target is the corpus's **spam versus ham** label; it is not a verified malware or phishing verdict.

The objective is to create an explainable and reproducible learning workflow, interpret its results, and demonstrate inference in a lightweight Streamlit application. Inputs remain local. There are no paid services or API dependencies.

## 2. Dataset and exploratory analysis

The data source is the [Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/), using the `20030228_spam`, `20030228_easy_ham`, and `20030228_hard_ham` archive files. The corpus archive category supplies each binary label. The parser handles MIME messages and derives the same numeric features for the training set and the prototype:

1. Message body character length and word count.
2. Count of HTTP/HTTPS and `www.` URLs.
3. MIME attachment count.
4. Subject indicators for urgency and account language.
5. Body indicators for urgency, verification, prizes/winnings, and passwords.
6. Sender indicators for no-reply naming and digits in the sender domain.
7. HTML presence, subject length, and body uppercase-letter ratio.

The resulting feature table contains 3,253 messages: 2,752 ham (84.6%) and 501 spam (15.4%). No missing labels or feature values were found. There are 64 repeated numeric feature rows; different emails can share these coarse measurements, so these are not treated as duplicate-message proof. The distributions are skewed: body length ranges from 0 to 202,500 characters and URL count from 0 to 3,134. Tukey IQR flags many upper-tail values; these were not removed because they can represent genuinely long messages or link-heavy spam, and the scale-sensitive model standardizes each feature.

`reports/data_profile.json` records per-feature summaries, IQR outlier counts, class-wise medians, and missing-value counts. `reports/class_distribution.png` and `reports/feature_distributions.png` visualize class balance and the feature distributions by label. The ham/spam median body lengths are 934.5 and 1,350 characters, respectively; the spam median uppercase ratio is about 9.2%, versus 5.5% for ham. These differences are descriptive associations, not causal findings.

## 3. Methodology

The dataset builder downloads the three source archives when they are not already present, parses each message without extracting archive paths to disk, and writes `data/processed/email_features.csv`. A SHA-256 prefix identifies each record without storing the full raw message in the derived table. The feature extraction module is shared by training and app inference to keep the input schema consistent.

The model is a scikit-learn pipeline: `StandardScaler` followed by class-weighted (`balanced`) Logistic Regression. Standardization is important because message length and URL counts have much larger numeric ranges than binary indicators. Class weighting addresses the corpus's imbalance without changing the reported test-set class proportions. A stratified 80/20 train/test split with `random_state=42` gives 2,602 training and 651 test records. The holdout is used only for final evaluation; no hyperparameter search or model selection uses it.

## 4. Evaluation

| Measure | Holdout result |
|---|---:|
| Accuracy | 78.2% |
| Spam precision | 38.0% |
| Spam recall | 66.0% |
| Spam F1 | 48.2% |
| ROC-AUC | 0.777 |

Confusion matrix, with actual classes in rows (ham, spam) and predicted classes in columns (ham, spam):

|  | Predicted ham | Predicted spam |
|---|---:|---:|
| Actual ham | 443 | 108 |
| Actual spam | 34 | 66 |

The model finds 66 of 100 spam messages but labels 108 of 551 ham messages as spam. Its 38% spam precision is a major warning: a message flagged by this model is more likely to be a false alarm than the precision value a practical quarantine workflow would require. Accuracy alone is misleading because ham is the majority class. The ROC-AUC indicates ranking signal, but does not remedy poor precision at the default threshold. `reports/metrics.json` contains full-precision values; the confusion-matrix plot and ROC plot are generated alongside the metrics.

## 5. Prototype and reproducibility

The Streamlit app accepts sender, subject, body, and attachment count, or parses a user-selected `.eml` file locally. It displays the model label, estimated spam score, and extracted indicators. The app surfaces a clear warning that the model uses old data and is not a security control.

To reproduce the result, follow the commands in the project README: install `requirements.txt`, run `python -m src.email_threat.dataset`, then `python -m src.email_threat.train`. The default split seed is 42. Run `python -m pytest -q` for parser, feature, and dataset tests. The trained artifact, holdout metrics, and figures are written to `models/` and `reports/`.

## 6. Limitations, future work, and conclusion

The 2002–2003 corpus predates current phishing campaigns and does not establish performance on present-day email, other languages, or different organizations. Spam is not synonymous with phishing or malware. Hand-engineered keywords are brittle, sender-domain digit patterns can be misleading, and a random holdout from one corpus does not measure temporal or external generalization. The dataset also has class imbalance, a low spam precision, and repeated coarse feature vectors. The prototype does not inspect link destinations, authenticate senders, or safely analyze attachments.

Future work should evaluate a temporally and organizationally separate corpus, add text features and sender-authentication context, compare Logistic Regression against at least one alternative with cross-validation confined to the training data, calibrate and select a decision threshold against explicit false-positive costs, and report confidence intervals. All work should preserve privacy and avoid deploying automatic blocking without a robust independent validation.

In conclusion, the project fulfills the prototype objective and provides a transparent dataset-to-demo workflow. The measured false-positive rate and historical data limitations mean its output should be used only for education and experimentation.

## References

1. Apache SpamAssassin, “Public Corpus,” <https://spamassassin.apache.org/old/publiccorpus/>. Dataset archives used: `20030228_spam`, `20030228_easy_ham`, and `20030228_hard_ham`.
2. scikit-learn developers, “LogisticRegression,” <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>.
3. scikit-learn developers, “Model evaluation: quantifying the quality of predictions,” <https://scikit-learn.org/stable/modules/model_evaluation.html>.

## Ten-day delivery plan

| Day | Milestone |
|---|---|
| 1 | Research, define spam/ham scope, and cite the data source |
| 2 | Download and parse the corpus |
| 3 | Audit class balance, missing values, repeated feature rows, and distributions |
| 4 | Build the interpretable Logistic Regression baseline |
| 5 | Evaluate with precision, recall, F1, ROC-AUC, and the confusion matrix |
| 6 | Review false positives, limitations, and next-step improvements |
| 7 | Implement the local Streamlit prototype |
| 8 | Integrate and run automated tests |
| 9 | Prepare the paper, figures, and reproducible outputs |
| 10 | Present the results and explain the design and limitations |
