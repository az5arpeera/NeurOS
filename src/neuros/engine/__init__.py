"""The online engine (architecture §7): headset samples in, intent events out.

``Engine`` runs the stages of §7.1 on each new window: shared preprocessing (the training
code), the headset's channels placed in the model's layout, the quality gate, online alignment
toward a population prior, the decoder and the decision policy. ``Simulator`` replays a
recording as the timestamped chunks a headset stream would deliver.

Transport (LSL in, WebSocket out) wraps this core and comes separately; everything here is
pure computation, so the end-to-end tests run without a network.
"""
