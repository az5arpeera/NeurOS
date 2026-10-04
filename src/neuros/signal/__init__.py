"""The causal signal chain shared by training and the live engine (architecture §3, §4.2).

    native-rate chunk ──► CausalFilter ──► TimestampResampler ──► 128 Hz stream ──► windows
                          notch + band-pass  linear interpolation   (CAR per window)
                          (state carried)    on sample timestamps

Train–serve parity (principle 1) is the whole point of this package: offline preprocessing is
the streaming code run over a whole recording in one chunk. Tests assert that any chunking of
the same samples gives bit-identical output, so a model can never be trained on a signal the
engine doesn't produce.
"""
