| Model | Split | Val acc | Val macro-F1 | Test acc | Test macro-F1 (95% CI) | Train time |
|---|---|---|---|---|---|---|
| MLP (TF-IDF) | temporal | 0.8685 | 0.8704 | 0.8738 | 0.8739 (0.8658 to 0.8812) | 22 s |
| MLP (TF-IDF) | random | 0.8836 | 0.8846 | 0.8840 | 0.8849 (0.8780 to 0.8923) | 16 s |
| BiLSTM | temporal | 0.8688 | 0.8711 | 0.8684 | 0.8689 (0.8609 to 0.8768) | 33 min |
| BiLSTM | random | 0.8816 | 0.8821 | 0.8812 | 0.8817 (0.8743 to 0.8889) | 22 min |
| DistilBERT | temporal | 0.8907 | 0.8941 | 0.8943 | 0.8945 (0.8873 to 0.9012) | 49 min |
| DistilBERT | random | 0.9026 | 0.9035 | 0.9046 | 0.9053 (0.8990 to 0.9112) | 37 min |
