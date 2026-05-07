# TrackNet Weights

Place the TrackNetV2 weights here if you want the app to find them automatically without a CLI flag.

Supported formats:

- yastrebksv/TrackNet PyTorch `state_dict` checkpoint
- TorchScript-exported TrackNet model

Expected filename:

```text
tracknetv2.torchscript.pt
```

Default path used by the app:

```text
/Users/james/playground/forespin/models/tracknet/tracknetv2.torchscript.pt
```

If that file is present, `forespin analyze ...` will use it automatically.
