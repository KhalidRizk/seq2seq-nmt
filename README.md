# Seq2Seq-NMT: Arabic to English Neural Machine Translation

This is a PyTorch implementation of a neural machine translation model for translating Arabic sentences to English using the Seq2Seq architecture with attention. The model is trained on the OPUS100 dataset, which consists of parallel text in Arabic and English.

## Data Preprocessing

Before training the model, the Arabic and English sentences were preprocessed using the following steps:

- Tokenization: Splitting the sentences into words or subwords
- Lowercased the English text to reduce vocabulary size and increase model accuracy.
- Removed diacritics from the Arabic text as they do not impact the meaning of the text and would increase the size of the vocabulary.
- Cleaning the text by removing certain characters and symbols
- Replacing some common abbreviations with their full forms


## Model Architecture
The Seq2Seq model consists of an encoder and a decoder. The encoder takes the input Arabic sentence and produces a fixed-length context vector, which is then used by the decoder to generate the output English sentence. The attention mechanism is used to allow the decoder to focus on different parts of the input sentence at each time step.

The architecture of the model is as follows:

```python
Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(37705, 256)
    (rnn): GRU(256, 512, bidirectional=True)
    (fc): Linear(in_features=1024, out_features=512, bias=True)
    (dropout): Dropout(p=0.3, inplace=False)
  )
  (decoder): Decoder(
    (attention): Attention(
      (attn): Linear(in_features=1536, out_features=512, bias=True)
      (v): Linear(in_features=512, out_features=1, bias=False)
    )
    (embedding): Embedding(18343, 256)
    (rnn): GRU(1280, 512)
    (fc_out): Linear(in_features=1792, out_features=18343, bias=True)
    (dropout): Dropout(p=0.3, inplace=False)
  )
)
```

## Training

The model was trained using the Adam optimizer. The loss function is cross-entropy loss. Early stopping is used to prevent overfitting. The model is trained for a maximum of 25 epochs with a batch size of 16.
 The model was trained for 6 epochs, and early stopping was applied if the validation loss did not improve for 3 consecutive epochs.

Here are the training results:
| **Epoch** | **Train Loss** | **Train PPL** | **Val. Loss** | **Val. PPL** |
|-------|------------|-----------|-----------|----------|
| 1     | 5.630      | 278.648   | 5.393     | 219.891  |
| 2     | 4.529      | 92.630    | 5.107     | 165.123  |
| **3**     | **3.859**      | **47.402**    | **5.102**     | **164.297** |
| 4     | 3.375      | 29.222    | 5.262     | 192.843  |
| 5     | 3.061      | 21.341    | 5.385     | 218.133  |
| 6     | 2.889      | 17.978    | 5.506     | 246.208  |

## Future Work

There are several possible improvements and extensions to this project, including:

- **Using a larger or more diverse dataset for training**: While the OPUS100 dataset used for training is a large parallel corpus, it may not capture all of the possible variations in Arabic or English text. Using additional data sources, such as web-crawled text or domain-specific corpora, could help improve the model's performance on different text types and domains.

- **Experimenting with different hyperparameters or model architectures**: There are many hyperparameters that can be tuned in a neural machine translation system, such as the number of layers in the encoder and decoder, the size of the hidden states, and the learning rate of the optimizer. Experimenting with different values for these hyperparameters, as well as trying different model architectures such as Transformers, could help improve the model's performance.

- **Implementing a beam search decoding algorithm for improved translation quality**: The current model uses a greedy decoding algorithm, which can lead to suboptimal translations. Implementing a beam search decoding algorithm, which explores multiple candidate translations at each time step, could help improve the overall translation quality.

- **Applying transfer learning or pretraining techniques to improve performance on low-resource settings**: Arabic is a low-resource language for machine translation, meaning that there are relatively few parallel corpora available for training. Transfer learning or pretraining techniques, such as multilingual models or unsupervised pretraining, could help improve the model's performance on low-resource settings.

## Contributing
Contributions to this project are welcome! If you find a bug, have a feature request, or want to contribute code, please open an issue or submit a pull request.

## Conclusion

In this project, we implemented a neural machine translation model for translating Arabic sentences to English using the Seq2Seq architecture with attention. The model was trained on the OPUS100 dataset and achieved reasonable translation performance. There are several possible ways to improve the model's performance, and we hope that this project can serve as a starting point for future research in Arabic-English machine translation.