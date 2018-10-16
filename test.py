import numpy as np
from tensorflow.contrib import learn

positive_examples = list(open("rt-polarity.pos", "r", encoding='utf-8').readlines())
positive_examples = [s.strip() for s in positive_examples]
negative_examples = list(open("rt-polarity.neg", "r", encoding='utf-8').readlines())
negative_examples = [s.strip() for s in negative_examples]

x_text = positive_examples + negative_examples

positive_labels = [[0, 1] for _ in positive_examples]
negative_labels = [[1, 0] for _ in negative_examples]

y = np.concatenate([positive_labels,negative_labels],axis=0)

max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

np.random.seed(10)

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(0.1 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


print(y_train)

# np.random.seed(10)


