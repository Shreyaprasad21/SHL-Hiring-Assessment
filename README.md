# 🗣️ SHL Grammar Scoring Engine

A machine learning solution developed for the SHL Hiring Assessment to evaluate the grammatical accuracy of spoken English from audio samples.

## 📌 Objective

To build a **Grammar Scoring Engine** that:
- Accepts an audio file (`.wav`) as input.
- Predicts a **continuous grammar score between 0 and 5**.
- Utilizes deep audio embeddings from OpenAI's **Whisper** model.
- Trains a regression model to evaluate spoken grammar accuracy.

## 📂 Dataset Structure

```
dataset/
├── audios_train/           # 444 training audio files
├── audios_test/            # 195 test audio files
├── train.csv               # Training metadata (filename, label)
├── test.csv                # Test metadata (filename)
├── sample_submission.csv   # Submission format template
```

## 📦 Requirements

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

**Required Libraries**:
- `whisper`
- `torch`, `torchaudio`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `scipy`
- `tqdm`

> Make sure your system has a **GPU** and CUDA if possible (Whisper benefits greatly from it).

## 🚀 How It Works

### 1. 📥 Preprocessing & Feature Extraction
- Load and pad/truncate each audio file using Whisper.
- Convert to **mel spectrogram** and extract deep embeddings using Whisper’s encoder.
- Average the embedding over time to create fixed-length feature vectors.

### 2. 🧠 Model Training
- Split training data into **train** and **validation**.
- Train a `MLPRegressor` with two hidden layers: `(512, 256)` and ReLU activations.

### 3. 📊 Evaluation
- Evaluate the model using **Pearson Correlation** on the validation set.
- Plot a scatter plot of predicted vs true grammar scores.

### 4. 🧪 Prediction & Submission
- Generate predictions for the test audio set.
- Save results in `submission.csv` matching the `sample_submission.csv` format.

## 📈 Evaluation Metric

- Primary metric: **Pearson Correlation Coefficient** between predicted and actual scores.

```python
📈 Pearson Correlation: 0.83  # (example output)
```

## 📊 Visualization

A scatter plot is generated to visualize predicted vs true scores:

![Grammar Score Scatter Plot](./assets/predicted_vs_true.png)  <!-- Add path if saving plots -->

## 🗃️ File Descriptions

| File                     | Description                                   |
|--------------------------|-----------------------------------------------|
| `train.csv`              | Contains audio file names + grammar scores.   |
| `test.csv`               | Contains test audio file names.               |
| `submission.csv`         | Final output with predicted scores.           |
| `sample_submission.csv`  | Template for formatting submission.           |
| `notebook.ipynb`         | Jupyter notebook with full pipeline.          |

## 📌 Results

| Metric              | Value         |
|---------------------|---------------|
| Pearson Correlation | ~0.83         |
| Test Predictions    | 195 Samples   |
| Output File         | `submission.csv` |

## 🔬 Future Improvements

- Use **larger Whisper models** (e.g., `medium`, `large`) for better embeddings.
- Integrate **data augmentation** (pitch shift, noise) to improve robustness.
- Try **transformer-based** regression models.
- Perform **hyperparameter tuning** for MLP.

## 👨‍💻 Author

- **Shreya Prasad**  
- Built as part of SHL Labs Hiring Assessment.

---
