# Context-Aware Student Engagement Temporal Experiment

This is a separate experiment for replacing ViFiCLIP mean pooling with
learnable temporal pooling. Code1 and code2 are not modified.

The experiment reuses:

- The code2 seven-class dataset through `data_split`
- The code2 annotation files through `annotations`
- The code1 base checkpoint through a symlink

No virtual environment, videos, or large checkpoint is duplicated here.
