# small-car

1. `pip install sherpa-onnx`
2. `pip install numpy`
3. `pip install librosa`
4. `pip install sounddevice`
5. `pip install sentence-transformers`
6. 下载 [sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2) 并解压到 `model/ASR/` 文件夹

Speed control:
- The voice control supports acceleration and deceleration commands in Chinese: `加速`, `减速`, and synonyms like `快点`, `慢一点`.
- The client maintains a `current_speed` (0–100) and sends `speed` with movement commands to the control server.
- If you're using the Raspberry Pi server, please update your server to accept `speed` in the JSON payload, e.g.:

```json
{
	"command": "FORWARD",
	"speed": 60
}
```

If you don't want a server-side change, the client will continue sending the `command` only (no `speed`) for basic commands. There is also a `SET_SPEED` command (sent when the vehicle is stopped and you say `加速`/`减速`) which you can choose to honor or ignore on the server.

Default behaviour:
- default `current_speed` = 50, `speed_step` = 10
- speed range 0..100

