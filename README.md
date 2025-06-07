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

📂 **Evaluation**: [View on Notebook](https://github.com/fc63/gender-classification/blob/main/Evaluate/modelv3.ipynb)

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

## 📂 Execution Order & Source Code

To reproduce the results, it is recommended to run the code in **Google Colab** and **mount your Google Drive**.  
You will need access to the `datasets/` and `models/` folders inside your Drive, which contain preprocessed `.pkl` files and trained checkpoints.  
If you don't have these, you can request them from the author.

The Jupyter notebooks in the [GitHub repository](https://github.com/fc63/gender-classification) are designed to be run in the following order:

1. **EuroParl Dataset Normalization**  
   ➤ [`europarl_normalized.ipynb`](https://github.com/fc63/gender-classification/blob/main/europarl_normalized/europarl_normalized.ipynb)

2. **Learning Rate Finder on Normalized EuroParl**  
   ➤ [`lrfinder.ipynb`](https://github.com/fc63/gender-classification/blob/main/lr_finder/lrfinder.ipynb)

3. **Training on Normalized Dataset (First Model)**  
   ➤ [`1.ipynb`](https://github.com/fc63/gender-classification/blob/main/gp_model_first_3_epoch/1.ipynb)

4. **Best model at step 24750 saved to Drive**

5. **Lehçe Dataset Creation**  
   ➤ [`lehce1.ipynb`](https://github.com/fc63/gender-classification/blob/main/lehce%20dataset/lehce1.ipynb)
   ➤ [`lehce dataset`](https://github.com/fc63/gender-classification/tree/main/lehce%20dataset) (the resulting dataset is here as pickle, but I changed the name. otherwise it is the same dataset.)

7. **Lehçe → English Translation**  
   ➤ [`lehce-eng.ipynb`](https://github.com/fc63/gender-classification/blob/main/pl%20to%20eng%20translate/lehce-eng.ipynb)

8. **Russian Dataset Creation**  
   ➤ [`rus_gender.ipynb`](https://github.com/fc63/gender-classification/blob/main/rus_gender/rus_gender.ipynb)

9. **Russian → English Translation**  
   ➤ [`1.ipynb`](https://github.com/fc63/gender-classification/blob/main/rus_translate/1.ipynb)

10. **NPTEL Dataset Preprocessing**  
   ➤ [`nptel.ipynb`](https://github.com/fc63/gender-classification/blob/main/nptel%20dataset/nptel.ipynb)

11. **Combining Lehçe + Russian + NPTEL**  
    ➤ [`1.ipynb`](https://github.com/fc63/gender-classification/blob/main/combined_3_datasets/1.ipynb)

12. **Blog Dataset (XML → Pickle)**  
    ➤ [`g_blogs.ipynb`](https://github.com/fc63/gender-classification/blob/main/g_blogs/g_blogs.ipynb)

13. **Blog Dataset Cleaning & Merging with 3 Datasets**  
    ➤ [`1.ipynb`](https://github.com/fc63/gender-classification/blob/main/combine_informal/1.ipynb)

14. **Merging EuroParl + Combined Informal Dataset**  
    ➤ [`1.ipynb`](https://github.com/fc63/gender-classification/blob/main/mergealldatasets/1.ipynb)

15. **Evaluation of Model Step 24750**  
    ➤ [`model24750.ipynb`](https://github.com/fc63/gender-classification/blob/main/Evaluate/model24750.ipynb)

16. **Phase 2: Fine-tuning on Merged Dataset**  
    ➤ [`1.ipynb`](https://github.com/fc63/gender-classification/blob/main/gpmodel_v3/1.ipynb)

17. **Evaluation of Fine-tuned Final Model (gp_modelv3)**  
    ➤ [`modelv3.ipynb`](https://github.com/fc63/gender-classification/blob/main/Evaluate/modelv3.ipynb)

🧠 **Note:** The final published model on Hugging Face is the one fine-tuned in step 15 and referred to as `gp_modelv3`.

---

## 👨‍🔬 Author & License

**Author**: Furkan Çoban  
**Project**: CENG-481 Gender Prediction Model  
**License**: MIT
