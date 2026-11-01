# 荷物積み込みロボットの階層制御: LLM計画器の自動最適化とDAgger方策

Physical AI応用1の最終課題です。空港の手荷物をコンテナに積み込む課題をMuJoCoで作り、上位の計画をLLM、下位の動作をニューラル方策が受け持つ2層の制御を学習させました。上位はDSPyでプロンプトを自動最適化し、下位は模倣学習(BC)にDAggerを重ねています。さらにGenesisで、同じ条件の模倣学習と強化学習(PPO)を比べました。

- 動画(YouTube、2 倍速、合成音声の解説つき): https://youtu.be/WB9PLGGRrfQ

## 課題

- 荷物7〜12個を、6か所の床スロットと2段までの積み重ねでコンテナに積み込む。
- 荷物には割れ物、重さ、大きさの属性があり、「割れ物の上に置けない」「重い物を軽い物の上に置けない」「大きい物を小さい物の上に置けない」という積み方のルールがある。このルールはLLMに教えない。
- 指示文には「割れ物は最後に」「重い荷物から先に」のような順序の要求が、英語か日本語で付くことがある。
- 許容誤差8mm。下位の評価では0.6cmの動作ノイズを加える。

## 仕組み

| 層 | 役割 | 使ったもの |
|---|---|---|
| 上位 | 指示文と荷物の一覧から計画(pick / place_slot / place_on の列)を書く | claude-haiku-5-5、DSPy3.4.1のMIPROv2(シミュレーターでの成功率を指標に、指示文と例を自動最適化) |
| 下位 | 計画の各スキルをアームの動きにする | MLP(13 → 4、幅 512、深さ 4)。BC で学習後、DAgger 4ラウンド(PyTorch、GPU) |
| 環境 | 物理と成否の判定 | MuJoCo 3.17.0。ガントリーと吸着の簡易モデル |
| 比較 | 模倣学習と強化学習の比較 | Genesis1.4.5(2048環境をGPUで並列実行)。BC、DAgger、PPO |
| 実行 | ノートブックの自動実行 | ModalのGPUコンテナ(NVIDIA A10、約 95 分) |

## リポジトリの構成

```
final_project_v7_modal.ipynb   本体のノートブック(環境、計画器、DSPy、BCとDAgger、Genesis)
run_v7_on_modal.py             Modalで上のノートブックを自動実行するスクリプト
genesis_part.py                GenesisでのBC/DAgger/PPO比較(ノートブックのセルと同じ内容)
MODAL_README.md                Modalでの実行手順
```

## 実行方法

ModalのGPUコンテナで実行します。API キーはModalのSecretにだけ置き、コードには書きません。

```bash
pip install modal
modal setup
modal secret create anthropic-secret ANTHROPIC_API_KEY=<Anthropic の API キー>
PAI_GPU=A10G modal run --detach run_v7_on_modal.py
```

結果は Modal Volumeに残り、modal_out/<実行名>/`に落ちます。詳しくは`MODAL_README.md`を見てください。LLMの呼び出しは約1,000回(claude-haiku-5-5)で、費用は 1.4 ドル前後でした。

