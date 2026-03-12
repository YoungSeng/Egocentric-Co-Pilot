
## Egocentric AI and Interactive Systems

Test on  NVIDIA GeForce RTX 4090

```
cd LLM-Orchestrated-Neuro-Symbolic-Execution
git clone https://github.com/SWivid/F5-TTS
cd F5-TTS
git checkout 12d6970271f5cdb91938f8ee7b2bbc60e60a0ea8
cd ..
```

Download `piece.pt` and `board.pt` and `model_1200000.safetensors` from [Baidu Netdisk](https://pan.baidu.com/s/1BqL2wdnbJaO880NKtBSQCg?pwd=wiwf) (and download pre-trained models from `SWivid/F5-TTS` in `huggingface`) and put them in `pre-trained-models`).

```
conda activate Egocentric-Co-Pilot

pip install websockets==12.0 opencv-python pillow torch matplotlib scikit-learn ultralytics transformers openai soundfile tomli cached-path omegaconf torchdiffeq torchaudio librosa x_transformers jieba pypinyin wandb accelerate ema-pytorch datasets pydub vocos sentence-transformers qwen-vl-utils torchcodec
pip install -U openai-whisper
CUDA_VISIBLE_DEVICES=1 python main.py
```

If you meet:

```
MoTTY X11 proxy: Authorisation not recognised

In case you are trying to start a graphical application with "sudo", read this article in order to avoid this issue:
https://blog.mobatek.net/post/how-to-keep-X11-display-after-su-or-sudo/

qt.qpa.xcb: could not connect to display localhost:10.0
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/mnt/dataX/ysc/Miniconda/envs/EgocentricCoPilot/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.

Aborted (core dumped)
```

Try to turn `visualization_mode` in `demo-Chinese-chess.py` to `False`.

You will see these output in back end.
```
2025-09-04 01:52:46,786 - INFO - server listening on 127.0.0.1:5000
2025-09-04 01:52:46,787 - INFO - WebSocket 服务器已在 ws://localhost:5000 启动
```

ssh -L 5000:127.0.0.1:5000 ysc@10.103.12.94

Then open `index.html` in front end.

```
ssh -L 5000:127.0.0.1:5000 ysc@10.103.12.94
```

<p align="center">
  <img src="screen shot.png" width="420" alt="App screenshot">
</p>
