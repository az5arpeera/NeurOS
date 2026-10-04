"""Alignment: per-person re-centring of the signal's spatial covariance (architecture §5).

Each person's (and each headset's) average spatial covariance differs, through electrode
placement, impedance, skull and amplitude scale. Euclidean Alignment (EA, He & Wu 2020) maps
every person's mean covariance to the identity, so a model sees everyone relative to their own
baseline. It's label-free and costs one small eigendecomposition.
"""
