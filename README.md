# Gender Prediction from Text ✍️ → 👩‍🦰👨

This model **predicts** the likely **gender** of an anonymous speaker or writer based solely on the content of an English text. It is built upon [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large) and fine-tuned on a diverse, multilingual, and multi-domain dataset with both formal and informal texts.

📍 **Space link**: [🔗 Try it out on Hugging Face Spaces](https://huggingface.co/spaces/fc63/Gender_Prediction)  
📁 **Model repo**: [🔗 View on Hugging Face Hub](https://huggingface.co/fc63/gender_prediction_model_from_text)  
🧠 **Source code**: [GitHub](https://github.com/fc63/gender-classification)

---

## 📊 Model Summary

- **Base model**: `microsoft/deberta-v3-large`
- **Fine-tuned on**: binary gender classification task (`female` vs `male`)
- **Best F1 Score**: `0.69` on a balanced multi-domain test set
- **Max token length**: 128
- **Evaluation Metrics**:
  - F1: 0.69
  - Accuracy: 0.69
  - Precision: 0.69
  - Recall: 0.69

---

## 🧾 Datasets Used

| Dataset | Domain | Type |
|--------|--------|------|
| [samzirbo/europarl.en-es.gendered](https://huggingface.co/datasets/samzirbo/europarl.en-es.gendered) | Formal speech (Parliament) | English |
| [czyzi0/luna-speech-dataset](https://huggingface.co/datasets/czyzi0/luna-speech-dataset) | Phone conversations | Polish → Translated |
| [czyzi0/pwr-azon-speech-dataset](https://huggingface.co/datasets/czyzi0/pwr-azon-speech-dataset) | Phone conversations | Polish → Translated |
| [sagteam/author_profiling](https://huggingface.co/datasets/sagteam/author_profiling) | Social posts | Russian → Translated |
| [kaushalgawri/nptel-en-tags-and-gender-v0](https://huggingface.co/datasets/kaushalgawri/nptel-en-tags-and-gender-v0) | Spoken transcripts | English |
| [Blog Authorship Corpus](https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) | Blog posts | English |

All datasets were normalized, translated if necessary, deduplicated, and **balanced via random undersampling** to ensure equal representation of both genders.

---

## 🛠️ Preprocessing & Training

- **Normalization**: Cleaned quotes, dashes, placeholders, noise, and HTML/code from all datasets.
- **Translation**: Used `Helsinki-NLP/opus-mt-*` models for Polish and Russian data.
- **Undersampling**: Random undersampling to balance male and female samples.
- **Training Strategy**:
  - LR Finder used to optimize learning rate (`2.66e-6`)
  - Fine-tuned using early stopping on both F1 and loss
  - Step-based evaluation every 250 steps
  - Best checkpoint at step 24,750 saved and evaluated
- **Second Phase Fine-tuning**:
  - Performed on full merged dataset for 2 epochs
  - Used cosine learning rate scheduler and warm-up steps

---

## 📈 Performance (on full merged test set)

| Class | Precision | Recall | F1-Score | Accuracy | Support |
|-----|-----|--------|----------|---------|---------|
| Female | 0.70 | 0.65 | 0.68 | | 591,027 |
| Male   | 0.68 | 0.72 | 0.70 | | 591,027 |
| **Macro Avg** | 0.69 | 0.69 | **0.69** | | 1,182,054 |
| **Accuracy**  |           |        | | **0.69** | 1,182,054 |

---

## 📦 Usage Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "fc63/gender_prediction_model_from_text"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to("cuda")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = round(probs[0][pred].item() * 100, 1)
    gender = "Female" if pred == 0 else "Male"
    return f"{gender} (Confidence: {confidence}%)"
```
```
sample_text = "I love writing in my journal every night. It helps me reflect on the day and plan for tomorrow."
print(predict(sample_text))
```
The Output Of This Sample:
```
Female (Confidence: 84.1%)
```

---

## 👨‍🔬 Author & License

**Author**: Furkan Çoban  
**Project**: CENG-481 Gender Prediction Model  
**License**: MIT
