I am working on a gender classification model that i think can be used for more than one objectives. For this, I am training a model using the [samzirbo/europarl.en-es.gendered](https://huggingface.co/datasets/samzirbo/europarl.en-es.gendered) dataset. The dataset is a completely structured dataset. I also use microsoft's deberta-v3-large transformers model for training.

The objectives for which I think it can be used:

Objective 1 is to identify common word patterns between the same genders that the human mind is not consciously aware of.

Objective 2 to identify differences between the issues prioritized by male and female politicians in the European Parliament.

Objective 3 Gender prediction for articles/text with unknown speaker and author.