# Experiment 703 — continuous five-epoch LoRA sweep

One fold-0 run per architecture and adapter rank. Each run trains continuously for
five epochs and writes a separate adapter, predictions, and validation report after
every epoch. The six arms differ only by architecture and the preregistered rsLoRA
rank/alpha pairs: 16/32, 32/64, and 64/128.
