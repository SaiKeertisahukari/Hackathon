# AI for Safer Online Spaces for Women

## Overview
This project was developed for the **"AI for Safer Online Spaces for Women"** hackathon, aimed at leveraging AI and NLP techniques to foster safer digital conversations. Our solution focuses on reconstructing conversations, classifying discussions, and detecting toxicity to create a more inclusive online environment.

## Dataset
- **Reference Paper**: [An Expert-Annotated Dataset for the Detection of Online Misogyny](https://github.com/ellamguest/online-misogyny-eacl2021)
- **Dataset Information**:
  - **Text Data**: Online discussions extracted from Reddit.
  - **Annotations**: Labels for toxicity, misogyny, and contextual information.
  - **Metadata**: Includes subreddit, author, timestamps, and classification labels.

## Tasks and Approach
We implemented solutions for four major tasks:

### Task 1: Parent-Child Conversation Reconstruction
**Objective**: Reconstruct fragmented online discussions to maintain context and improve moderation.

**Approach**:
- Used NLP techniques to analyze conversation flow.
- Implemented sequence-based models to predict missing context.
- Trained a custom BERT model for summarization.

**Model**:
- Utilized transformer-based models (BART, GPT-based) for summarization.
- Evaluated using BLEU, ROUGE, Perplexity, and Semantic Similarity.

### Task 2: Subreddit-Based Topic Classification
**Objective**: Classify discussions based on their respective subreddits.

**Approach**:
- Preprocessed text data (removal of stop words, stemming, and tokenization).
- Fine-tuned pre-trained models such as RoBERTa for Classification.

**Evaluation**:
- Measured accuracy, precision, recall, and F1-score.
- Developed an interactive visualization to track topic trends.
- Calculated topic coherence scores to evaluate topic modeling performance.

### Task 3: Detecting Toxic or Harmful Comments
**Objective**: Identify and mitigate toxic and misogynistic content in online discussions.

**Approach**:
- Fine-tuned pre-trained models such as RoBERTa.
- Used AUC-ROC, AUC-PR, and confusion matrix for evaluation.
- Applied class balancing techniques.

**Results**:
- Achieved high classification accuracy on the test set.
- Evaluated model fairness using false positive/negative rates.

### Task 4: Context-Aware Misogyny Detection
**Objective**: Detect misogynistic language while considering context.

**Approach**:
- Fine-tuned BERT for Sequence Classification.
- Performed gendered word bias checks.
- Used LIME and SHAP for model explainability.


**Potential Use in Monitoring Systems**:
- The highlighted words from LIME and SHAP can be integrated into real-time monitoring systems to flag users for review.

## Model Training & Evaluation
Each model was trained using the following pipeline:

- **Data Preprocessing**: Tokenization, lowercasing, and removal of stop words.
- **Feature Extraction**: TF-IDF, word embeddings, and contextual embeddings.
- **Model Selection**: Traditional ML and deep learning models.
- **Hyperparameter Tuning**: Grid search and Bayesian optimization.

### Evaluation Metrics
| Task | Metrics |
|------|---------|
| Conversation Reconstruction | BLEU, ROUGE, Perplexity, Semantic Similarity |
| Topic Classification | Accuracy, Precision, Recall, F1-score, Topic Coherence Score |
| Toxicity Detection | AUC-ROC, AUC-PR, Confusion Matrix |
| Misogyny Detection | Accuracy, Precision, Recall, F1-score, Cohen’s Kappa Score |

## Challenges Faced and Solutions Implemented
1. **Data Imbalance in Toxicity Detection**
   - **Challenge**: The dataset contained significantly fewer misogynistic/toxic samples.
   - **Solution**: Applied class weighting and oversampling techniques.

2. **Maintaining Context in Conversation Reconstruction**
   - **Challenge**: Parent-child relationships in Reddit discussions were sometimes ambiguous.
   - **Solution**: Used BART’s sequence-to-sequence modeling to improve contextual consistency.

3. **Overfitting in Topic Classification**
   - **Challenge**: Some subreddits had very distinct vocabularies, leading to model overfitting.
   - **Solution**: Regularized models using dropout, data augmentation, and cross-validation.

4. **Bias in Misogyny Detection**
   - **Challenge**: Potential bias in classifying gender-related words.
   - **Solution**: Implemented gendered word bias checks and explainability techniques (LIME, SHAP).

## How to Run the Code
### Prerequisites
- **Python 3.8+**
- **Jupyter Notebook**
- **Required Libraries**:
  ```sh
  pip install transformers nltk torch datasets pandas numpy matplotlib seaborn scikit-learn gensim plotly wordcloud shap lime tqdm
  ```

### Steps
1. Clone the repository:
   ```sh
   git clone <repo_link>
   cd <repo_folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
4. Open and execute the notebooks

