"""ModalのGPUコンテナでfinal_project_v7_modal.ipynbを先頭から最後まで実行し、出力を手元へ持ち帰る。

準備 (1 回だけ):
    pip install modal
    modal setup
    modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...   # キーはここにだけ置く (このファイルには書かない)
実行(必ず--detachを付ける。付けないと、手元の端末が閉じたりPCがスリープしたりした時点で実行が止まる):
    modal run --detach run_v7_on_modal.py                  # 全章。Genesis(11.5 章) は別のGPUコンテナで同時に走らせ、ノートブックが 11.5 章で結果を受け取る。3時間前後
    modal run --detach run_v7_on_modal.py --no-parallel-genesis   #Genesisをノートブックの中で順に実行する(従来どおり。3〜4時間)
    modal run --detach run_v7_on_modal.py --skip-genesis   # 11.5 章(Genesis) を飛ばす
    PAI_GPU=A10G modal run --detach run_v7_on_modal.py     # GPUを変える(既定A100。A10G/L4/L40S/H100など)
途中経過:
    端末を開いたままなら、そのまま各セルの出力が流れる。閉じたあとは modal app logs pai-final-v7 で見られる。
出力:
    Modal Volume `pai-final-outputs` の <実行名>/ に残る。端末を開いたまま終われば、手元のmodal_out/<実行名>/へ自動でダウンロードされる。
    端末を閉じた場合や途中で止めた場合は、次のどちらかで取り出す:
        modal run run_v7_on_modal.py --download-only <実行名>
        modal volume get pai-final-outputs <実行名> ./modal_out/<実行名>
    Genesisを別コンテナで走らせたときは、その出力(genesis_results.json, genesis_clips.mp4, genesis_compare.png, genesis_log.txt)が<実行名>/genesis/にも残る。
    初回のイメージ作成ログに'genesis install failed'が出ていたら、Genesisの実行前に入れ直す(数分の追加)。
"""
import os
import pathlib
import time

import modal

NOTEBOOK = "final_project_v7_modal.ipynb"
VOLUME_NAME = "pai-final-outputs"
SECRET_NAME = "anthropic-secret"
OUT = pathlib.Path("/out")
HOURS = 60 * 60
HERE = pathlib.Path(__file__).resolve().parent             # ノートブックはこのファイルと同じフォルダから取る (modal run を実行した場所に依存しない)
GPU = os.environ.get("PAI_GPU", "A100")      # 手元の環境変数で選ぶ (modal run の前に PAI_GPU=L4 など)
GENESIS_GPU = os.environ.get("PAI_GENESIS_GPU", GPU)   # Genesis用コンテナのGPU(既定はノートブックと同じ)
GENESIS_ARGS = ("2048", "250")               # genesis_part.pyの引数(並列環境数,PPOの反復回数)。ノートブックの 11.5 章と同じ
GENESIS_FILES = ("genesis_results.json", "genesis_clips.mp4", "genesis_compare.png")   # ノートブックの 12 章(動画)と 13 章が読むファイル

image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")   # Debian12。debian_slimはワークスペースのImage Builder Versionが古いと Debian11(配布終了でaptが404)になるため使わない
    .apt_install("ffmpeg", "libegl1", "libgl1", "libegl-mesa0", "libgl1-mesa-dri", "libosmesa6", "libglib2.0-0", "git")   # EGL(mesa)とOSMesaの両方を入れ、ノートブック側で動く方を選ぶ(Debian12のパッケージ名)
    .pip_install("mujoco==3.17.0", "dspy[optuna]==3.4.1", "mediapy", "opencv-python-headless", "torch",
                 "matplotlib", "pandas", "papermill", "ipykernel", "jupyter_client")
    .run_commands("python -m venv --system-site-packages /opt/gvenv && /opt/gvenv/bin/pip install -q genesis-world || echo 'genesis install failed (notebook will retry at runtime)'")   #Genesisは別の仮想環境(本体のtorch/numpyと干渉させない)。失敗してもイメージ作成は止めない
    .env({"ON_MODAL": "1", "PYOPENGL_PLATFORM": "egl"})
    .add_local_file(HERE / NOTEBOOK, f"/root/{NOTEBOOK}")
)
app = modal.App("pai-final-v7", image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

KEEP = ("*.ipynb", "*.mp4", "*.png", "*.json", "*.csv", "*.pt", "*.py")


def _run(run_name: str, skip_genesis: bool, genesis_call_id: str = "", genesis_wait_min: int = 90):
    import glob, os, shutil, subprocess
    work = OUT / run_name
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)                                   # 出力はすべてVolume上に書く(途中で落ちても残る)
    shutil.copy(f"/root/{NOTEBOOK}", NOTEBOOK)
    env = dict(os.environ, SKIP_GENESIS="1" if skip_genesis else "0", PYTHONUNBUFFERED="1",
               GENESIS_CALL_ID=genesis_call_id, GENESIS_WAIT_MIN=str(genesis_wait_min))   # 11.5章が別コンテナのGenesisを待つときに使う
    print(f"== {run_name}: papermill {NOTEBOOK} (skip_genesis={skip_genesis}, genesis_call_id={genesis_call_id or 'なし'}) ==", flush=True)
    t0 = time.time()
    r = subprocess.run(["papermill", NOTEBOOK, f"results_{NOTEBOOK}", "--log-output", "--kernel", "python3",
                        "--request-save-on-cell-execute"], env=env)
    vol.commit()
    files = sorted(f for pat in KEEP for f in glob.glob(pat))
    print(f"== 終了 (returncode={r.returncode}, {(time.time() - t0) / 60:.0f} 分) | 保存: {files}", flush=True)
    leaked = find_key_leaks(".", os.environ["ANTHROPIC_API_KEY"])          # 結果ノートブックを含む全ファイルをAPIキーで走査する(公開前の確認)
    if leaked:
        raise RuntimeError(f"API キーが含まれているファイルがあります。公開しないでください: {leaked}")
    print("API キーの混入なし:", len(files), "ファイルを確認", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"papermill が失敗しました。results_{NOTEBOOK} の最後のセルにエラーが記録されています")


def find_key_leaks(folder, key):
    """folder 直下のファイルのうち、key (API キー) をそのまま含むものを返す。"""
    import os
    k = key.encode()
    return sorted(f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and k in open(os.path.join(folder, f), "rb").read())


@app.function(
    gpu=GPU,
    cpu=8.0,                 # MuJoCoの物理とMIPROv2(6スレッド)はCPUで動く。既定(0.125コア)のままだと空きのないワーカーで極端に遅くなる
    memory=16384,            
    timeout=8 * HOURS,
    secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["ANTHROPIC_API_KEY"])],
    volumes={OUT.as_posix(): vol},
)
def run_notebook(run_name: str, skip_genesis: bool, genesis_call_id: str = "", genesis_wait_min: int = 90):
    _run(run_name, skip_genesis, genesis_call_id, genesis_wait_min)


# ----------Genesis(11.5 章)を別のGPUコンテナで走らせる ----------
def _genesis_source() -> str:
    """ノートブックの `%%writefile genesis_part.py` セルから、スクリプト本体を取り出す (ファイルを二重に持たない)"""
    import json
    nb = json.load(open(f"/root/{NOTEBOOK}", encoding="utf-8"))
    for c in nb["cells"]:
        src = "".join(c["source"])
        if src.startswith("%%writefile genesis_part.py"):
            return src.split("\n", 1)[1]
    raise RuntimeError("ノートブックに %%writefile genesis_part.py のセルがありません")


def _genesis_python() -> str:
    """Genesis 用の Python (別 venv)。イメージ作成時に入らなかったときは ~/gvenv に入れる (ノートブックの 11.5 章と同じ手順)"""
    import os, subprocess, sys
    for p in ("/opt/gvenv/bin/python", os.path.expanduser("~/gvenv/bin/python")):
        if os.path.exists(p) and subprocess.run([p, "-c", "import genesis"], capture_output=True).returncode == 0:
            return p
    py = os.path.expanduser("~/gvenv/bin/python")
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", os.path.dirname(os.path.dirname(py))], check=True)
    subprocess.run([py, "-m", "pip", "-q", "install", "genesis-world"], check=True)
    subprocess.run([py, "-c", "import genesis; print('genesis', genesis.__version__)"], check=True)
    return py


def _genesis(run_name: str) -> dict:
    """<実行名>/genesis/ で genesis_part.py を実行し、結果ファイルを Volume に残しつつ、呼び出し側 (ノートブック) にも bytes で返す。
    失敗しても例外にせず ok=False で返す (ノートブックは Genesis の場面なしで続行する)"""
    import os, pathlib, subprocess, traceback
    work = OUT / run_name / "genesis"
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    t0 = time.time()
    res = dict(ok=False, returncode=None, minutes=0.0, log_tail="", files={})
    log_lines = []
    try:
        pathlib.Path("genesis_part.py").write_text(_genesis_source(), encoding="utf-8")
        py = _genesis_python()
        print(f"== {run_name}: Genesis を {py} で実行 (args={GENESIS_ARGS}) ==", flush=True)
        p = subprocess.Popen([py, "-u", "genesis_part.py", *GENESIS_ARGS], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=dict(os.environ, PYOPENGL_PLATFORM="egl"))
        with open("genesis_log.txt", "w") as log:
            for line in p.stdout:
                log.write(line)
                if line.strip() and not any(k in line for k in ("libEGL", "DeprecationWarning", "kwarg_node", "return index")):
                    print(line, end="", flush=True)
                    log_lines.append(line)
        p.wait()
        res["returncode"] = p.returncode
        for name in GENESIS_FILES:
            if os.path.exists(name):
                res["files"][name] = open(name, "rb").read()
        res["ok"] = p.returncode == 0 and "genesis_results.json" in res["files"]
    except Exception:
        log_lines.append(traceback.format_exc())
        print(log_lines[-1], flush=True)
    res["minutes"] = (time.time() - t0) / 60
    res["log_tail"] = "".join(log_lines[-15:])
    vol.commit()
    print(f"== Genesis 終了 (ok={res['ok']}, returncode={res['returncode']}, {res['minutes']:.1f} 分, files={sorted(res['files'])}) ==", flush=True)
    return res


@app.function(
    gpu=GENESIS_GPU,
    cpu=4.0,
    memory=16384,
    timeout=3 * HOURS,
    volumes={OUT.as_posix(): vol},
)
def run_genesis(run_name: str) -> dict:
    return _genesis(run_name)


def _download(run_name: str, dest: pathlib.Path):
    """Volumeのrun_name/以下をdest/に取り出す。entry.pathは先頭の/の有無どちらでも扱う。"""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in vol.listdir(run_name, recursive=True):
        rel = entry.path.lstrip("/")
        if entry.type != modal.volume.FileEntryType.FILE or rel == run_name:
            continue
        local = dest / pathlib.Path(rel).relative_to(run_name)
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        n += 1
    return n


def _download_or_explain(run_name: str):
    dest = pathlib.Path("modal_out") / run_name
    try:
        n = _download(run_name, dest)
        print(f"downloaded {n} files -> {dest}/  (results_{NOTEBOOK}, final_demo.mp4, *.png, report_numbers.json, ...)")
    except Exception as e:                                   # ダウンロードの失敗で、本体の例外 (papermill の失敗など) を隠さない
        print(f"ダウンロードに失敗しました: {e!r}\n手動で取り出す: modal volume get {VOLUME_NAME} {run_name} ./{dest}")


@app.local_entrypoint()
def main(skip_genesis: bool = False, parallel_genesis: bool = True, genesis_wait_min: int = 90, download_only: str = ""):
    """download_onlyに実行名を渡すと、実行せずにその出力だけを取り出す。
    parallel_genesis(既定True)ならGenesisを別のGPUコンテナで同時に走らせ、ノートブックは11.5章でその結果を最大genesis_wait_min分待つ"""
    if download_only:
        _download_or_explain(download_only)
        return
    run_name = time.strftime("run_%Y%m%d_%H%M")
    mode = "skip" if skip_genesis else ("parallel" if parallel_genesis else "inline")
    print(f"run name: {run_name}  (GPU {GPU}, genesis={mode})  | 取り出し: modal run run_v7_on_modal.py --download-only {run_name}")
    gcall = None
    try:
        if mode == "parallel":
            gcall = run_genesis.spawn(run_name)
            print(f"Genesis を別コンテナ (GPU {GENESIS_GPU}) で開始: call id {gcall.object_id}")
        run_notebook.remote(run_name, skip_genesis, gcall.object_id if gcall else "", genesis_wait_min)
    finally:
        if gcall is not None:
            try:
                g = gcall.get(timeout=60)
                print(f"Genesis コンテナ: ok={g['ok']} returncode={g['returncode']} {g['minutes']:.1f} 分")
            except Exception as e:
                print(f"Genesis コンテナの結果を取れませんでした: {e!r}")
        _download_or_explain(run_name)
