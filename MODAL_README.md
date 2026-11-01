# Modalでv7を実行する手順

ColabのGPU混雑と切断を避けるため、`final_project_v7_modal.ipynb`はModalのGPUコンテナで自動実行します。ブラウザを閉じても実行は続きます。

## 1回だけの準備(手元のPC)
```bash
pip install modal
modal setup                                                   # ブラウザでModalにログイン
modal secret create anthropic-secret ANTHROPIC_API_KEY=sk-ant-...   # APIキーはSecretにだけ置く
```
`final_project_v7_modal.ipynb`と`run_v7_on_modal.py`を同じフォルダに置きます。

## 実行
**必ず`--detach`を付けます。**付けないと、端末を閉じたりPCがスリープしたりした時点でModal側の実行も止まります(Colabの切断と同じことが起きます)。
```bash
cd "<このフォルダ>"
modal run --detach run_v7_on_modal.py                    # 全章(Genesisを含む)。目安3〜4時間、GPUはA100
modal run --detach run_v7_on_modal.py --skip-genesis     # 11.5章(Genesis)を飛ばす
PAI_GPU=A10G modal run --detach run_v7_on_modal.py       # GPUを変える(既定 A100。A10G/L4/L40S/H100など)
```
初回はイメージの作成(パッケージの導入)に10分前後かかります。2回目からは省かれます。
初回のビルドログの最後に`genesis install failed`と出ていたら、11.5章が実行時にGenesisを入れ直します(数分の追加)。
実行名(`run_20261101_1200`の形)は開始直後に表示されます。控えておいてください。
画面には、ノートブックの各セルの出力がそのまま流れます。端末を閉じたあとは次で見られます。
```bash
modal app logs pai-final-v7
```

## 出力
端末を開いたまま終われば、`modal_out/<実行名>/`に次が入ります(Colabのときと同じファイル)。
- `results_final_project_v7_modal.ipynb`(出力つきのノートブック。提出用)
- `final_demo.mp4`、`dagger_curve.png`、`genesis_compare.png`、`genesis_clips.mp4`
- `report_numbers.json`、`results_by_kind.csv`、`optimized_planner.json`、`policy.pt`、`genesis_results.json`

端末を閉じていた場合や、途中で止まった場合も、そこまでの出力はModal Volumeに残っています。実行名を指定して取り出します。
```bash
modal run run_v7_on_modal.py --download-only run_20261101_1200          # 実行名を指定して取り出す
modal volume get pai-final-outputs run_20261101_1200 ./modal_out/run_20261101_1200   # 上がだめなときの手動の取り出し
modal volume ls pai-final-outputs                                        # 実行名の一覧
```
終了時にランナーが、結果ノートブックを含む全ファイルにAPIキーが混入していないかを走査します。混入があれば例外で止まり、ファイル名が表示されます。

## 注意
- 費用はModalのGPU時間(A100 40GBは1時間あたり約2.1ドル、A10は約1.1ドル。2026-11-01の料金ページの秒単価から換算)と、Anthropic APIの費用(実績で4.5ドル)です。3.5時間なら GPU 代は7〜8ドルの見込みです。CPU8コアとメモリ16GiBの分が別に1時間あたり約0.5ドルかかります。
- `run_v7_on_modal.py`にはAPIキーを書きません。
- ノートブックは途中で止めて判断する「ゲート」(7章のベースラインが20%未満なら`SHOW_RULES = True`)を通らず最後まで進みます。結果ノートブックの7章を見て、必要なら設定を変えて再実行してください。

## 実行が途中で止まったとき

- Volumeの`<実行名>/`には、止まるまでに書かれたファイルと、エラーのセルまで出力が入った`results_final_project_v7_modal.ipynb`が残ります。どのセルで止まったかはこのノートブックで確認できます。
- MIPROv2の指示文提案でSonnetが空応答、または途中で切れた応答を返して止まる場合は、提案モデルをHaikuに切り替えて続行します(どちらを使ったかは`report_numbers.json`の`models.prompt`に残ります)。再試行の回数は`check_budget()`の出力に出ます。
- 下位方策の学習が終わった時点で`policy_history.json`を保存するので、その後で止まっても学習結果の数値は残ります。

## Genesis を別コンテナで同時に走らせる(既定)

- `modal run --detach run_v7_on_modal.py`は、先にGenesis用のGPUコンテナ(`run_genesis`)を起動し、同時にノートブックのコンテナを起動します。ノートブックは11.5章に着いたとき(開始から2時間弱)にGenesisの結果(`genesis_results.json`、`genesis_clips.mp4`、`genesis_compare.png`)を受け取り、動画と`report_numbers.json`に組み込みます。
- Genesisが先に終わっていなければ、11.5章で最大90分待ちます(`--genesis-wait-min 120`で変更)。上限を過ぎたときやGenesisが失敗したときは、Genesisの場面なしで先に進みます。
- Genesis側の出力とログはVolumeの`<実行名>/genesis/`にも残ります(`genesis_log.txt`を含む)。手元には`modal_out/<実行名>/genesis/`に落ちます。
- GPUを2台同時に使うので、Genesisの実行時間ぶん(A10で1時間なら約1.1ドル)の費用が増えます。Genesis側だけGPUを変えるには`PAI_GENESIS_GPU=L4`のように指定します。
- 従来どおりノートブックの中で順に実行するには`--no-parallel-genesis`を付けます。
