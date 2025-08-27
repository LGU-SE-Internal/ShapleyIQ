
```bash
sudo juicefs mount redis://10.10.10.119:6379/1 /mnt/jfs -d --cache-size=1024

mkdir data
cd data
ln -s /mnt/jfs/rcabench_dataset ./
ln -s /mnt/jfs/rcabench-platform-v2 ./
```

```sh
export RCABENCH_BASE_URL=http://10.10.10.220:32080
export RCABENCH_USERNAME=admin
export RCABENCH_PASSWORD=admin123
sudo -E .venv/bin/python run.py batch-test
```

```sh
sudo -E .venv/bin/python ./main.py eval single microrca rcabench_filtered ts0-ts-auth-service-stress-jv8m9r --clear
```
```
┌───────────────────┬───────────┬───────┬───────┬─────────────────────┬──────────┬────────────┬────────────┬────────────┬──────────┬──────────┬─────────┐
│ dataset           ┆ algorithm ┆ total ┆ error ┆ runtime.seconds:avg ┆      MRR ┆ AC@1.count ┆ AC@3.count ┆ AC@5.count ┆     AC@1 ┆     AC@3 ┆    AC@5 │
│ ---               ┆ ---       ┆   --- ┆   --- ┆                 --- ┆      --- ┆        --- ┆        --- ┆        --- ┆      --- ┆      --- ┆     --- │
│ str               ┆ str       ┆   u32 ┆   u32 ┆                 f64 ┆      f64 ┆        f64 ┆        f64 ┆        f64 ┆      f64 ┆      f64 ┆     f64 │
╞═══════════════════╪═══════════╪═══════╪═══════╪═════════════════════╪══════════╪════════════╪════════════╪════════════╪══════════╪══════════╪═════════╡
│ rcabench_filtered ┆ shapleyiq ┆   438 ┆     0 ┆           29.226472 ┆ 0.334031 ┆       81.0 ┆      146.0 ┆      232.0 ┆ 0.184932 ┆ 0.333333 ┆ 0.52968 │
└───────────────────┴───────────┴───────┴───────┴─────────────────────┴──────────┴────────────┴────────────┴────────────┴──────────┴──────────┴─────────┘
```
我们的系统的 trace 加了 loadgenerator 和 caddy，这两个入口和前端导致整个他俩基本上都排前二

```sh
sudo -E ./.venv/bin/python ./main.py eval batch -a shapleyiq -a ton -a microrank -a microhecl -a microrca -d rcabench_filtered --clear
```

```
│ dataset           ┆ algorithm ┆ total ┆ error ┆ runtime.seconds:avg ┆      MRR ┆ AC@1.count ┆ AC@3.count ┆ AC@5.count ┆     AC@1 ┆     AC@3 ┆     AC@5 │
│ ---               ┆ ---       ┆   --- ┆   --- ┆                 --- ┆      --- ┆        --- ┆        --- ┆        --- ┆      --- ┆      --- ┆      --- │
│ str               ┆ str       ┆   u32 ┆   u32 ┆                 f64 ┆      f64 ┆        f64 ┆        f64 ┆        f64 ┆      f64 ┆      f64 ┆      f64 │
╞═══════════════════╪═══════════╪═══════╪═══════╪═════════════════════╪══════════╪════════════╪════════════╪════════════╪══════════╪══════════╪══════════╡
│ rcabench_filtered ┆ microhecl ┆   438 ┆     0 ┆            6.767625 ┆      1.0 ┆       22.0 ┆       22.0 ┆       22.0 ┆ 0.050228 ┆ 0.050228 ┆ 0.050228 │
│ rcabench_filtered ┆ microrank ┆   438 ┆     0 ┆             9.28409 ┆  0.25329 ┆       70.0 ┆      101.0 ┆      121.0 ┆ 0.159817 ┆ 0.230594 ┆ 0.276256 │
│ rcabench_filtered ┆ microrca  ┆   438 ┆     0 ┆            7.363791 ┆ 0.163675 ┆        3.0 ┆       92.0 ┆      111.0 ┆ 0.006849 ┆ 0.210046 ┆ 0.253425 │
│ rcabench_filtered ┆ shapleyiq ┆   438 ┆     0 ┆           15.401277 ┆ 0.347455 ┆       83.0 ┆      156.0 ┆      239.0 ┆ 0.189498 ┆ 0.356164 ┆ 0.545662 │
│ rcabench_filtered ┆ ton       ┆   438 ┆     0 ┆            7.157751 ┆ 0.321249 ┆       77.0 ┆      150.0 ┆      202.0 ┆ 0.175799 ┆ 0.342466 ┆ 0.461187 │
```
去掉loadgenerator之后效果依然一般，但是shapleyiq确实已经是这几个里面最好的了，只能说我们的异常扩散确实狠？microhecl的效果很差




```
┌───────────────────┬───────────┬───────┬───────┬─────────────────────┬──────────┬────────────┬────────────┬────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ dataset           ┆ algorithm ┆ total ┆ error ┆ runtime.seconds:avg ┆      MRR ┆ AC@1.count ┆ AC@3.count ┆ AC@5.count ┆     AC@1 ┆     AC@3 ┆     AC@5 ┆    Avg@3 ┆    Avg@5 │
│ ---               ┆ ---       ┆   --- ┆   --- ┆                 --- ┆      --- ┆        --- ┆        --- ┆        --- ┆      --- ┆      --- ┆      --- ┆      --- ┆      --- │
│ str               ┆ str       ┆   u32 ┆   u32 ┆                 f64 ┆      f64 ┆        f64 ┆        f64 ┆        f64 ┆      f64 ┆      f64 ┆      f64 ┆      f64 ┆      f64 │
╞═══════════════════╪═══════════╪═══════╪═══════╪═════════════════════╪══════════╪════════════╪════════════╪════════════╪══════════╪══════════╪══════════╪══════════╪══════════╡
│ rcabench_filtered ┆ microhecl ┆  1124 ┆     0 ┆            7.195779 ┆      1.0 ┆      361.0 ┆      361.0 ┆      361.0 ┆ 0.321174 ┆ 0.321174 ┆ 0.321174 ┆ 0.321174 ┆ 0.321174 │
│ rcabench_filtered ┆ microrank ┆  1124 ┆     0 ┆             7.71865 ┆ 0.195632 ┆       46.0 ┆      260.0 ┆      330.0 ┆ 0.040925 ┆ 0.231317 ┆ 0.293594 ┆ 0.152432 ┆ 0.203737 │
│ rcabench_filtered ┆ microrca  ┆  1124 ┆     0 ┆             8.70987 ┆ 0.536379 ┆      386.0 ┆      577.0 ┆      755.0 ┆ 0.343416 ┆ 0.513345 ┆ 0.671708 ┆ 0.422005 ┆ 0.512277 │
│ rcabench_filtered ┆ shapleyiq ┆  1124 ┆     0 ┆           16.385051 ┆ 0.494786 ┆      412.0 ┆      581.0 ┆      724.0 ┆ 0.366548 ┆ 0.516904 ┆ 0.644128 ┆ 0.443654 ┆   0.5121 │
│ rcabench_filtered ┆ ton       ┆  1124 ┆     0 ┆           44.873549 ┆ 0.337656 ┆      229.0 ┆      390.0 ┆      540.0 ┆ 0.203737 ┆ 0.346975 ┆ 0.480427 ┆ 0.266607 ┆  0.33968 │
└───────────────────┴───────────┴───────┴───────┴─────────────────────┴──────────┴────────────┴────────────┴────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```
给 ton microrca和microhecl加上alarm node（from detector），以及修复了ton和microrca的metric加载之后，效果明显提升


```sh
export RCABENCH_BASE_URL=http://10.10.10.220:32080
export RCABENCH_USERNAME=admin
export RCABENCH_PASSWORD=admin123
```
```sh
# upload algorithm
bash deploy_all_algos.sh
```
```sh
# batch test
sudo -E .venv/bin/python run.py batch-test --label 8.23shapleyiqbatch
rca cross-dataset-metrics  -d pair-diag -dv all-8.23 --tag 8.23shapleyiqbatch  -a microrank -a ton  -a microhecl -a shapleyiq  -a microrca  -l service
rca cross-dataset-metrics  -d pair-diag -dv all-8.23 --tag 8.23microdig820c2d7 -a microdig -l service

```

```
┌───────────┬───────────┬─────────────────┬─────────┬───────┬───────┬───────┬───────┬───────┬──────┬───────┬─────┬─────┬─────┬────────────┬────────────────┐
│ algorithm ┆ dataset   ┆ dataset_version ┆ level   ┆  top1 ┆  top3 ┆  top5 ┆  avg3 ┆  avg5 ┆ time ┆   mrr ┆ as1 ┆ as3 ┆ as5 ┆ efficiency ┆ datapack_count │
│ ---       ┆ ---       ┆ ---             ┆ ---     ┆   --- ┆   --- ┆   --- ┆   --- ┆   --- ┆  --- ┆   --- ┆ --- ┆ --- ┆ --- ┆        --- ┆            --- │
│ str       ┆ str       ┆ str             ┆ str     ┆   f64 ┆   f64 ┆   f64 ┆   f64 ┆   f64 ┆  f64 ┆   f64 ┆ f64 ┆ f64 ┆ f64 ┆        f64 ┆            i64 │
╞═══════════╪═══════════╪═════════════════╪═════════╪═══════╪═══════╪═══════╪═══════╪═══════╪══════╪═══════╪═════╪═════╪═════╪════════════╪════════════════╡
│ microrank ┆ pair-diag ┆ all-8.23        ┆ service ┆ 0.139 ┆ 0.347 ┆ 0.477 ┆ 0.641 ┆ 1.255 ┆  0.0 ┆ 0.311 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1129 │
│ ton       ┆ pair-diag ┆ all-8.23        ┆ service ┆ 0.182 ┆ 0.312 ┆ 0.435 ┆ 0.562 ┆ 1.179 ┆  0.0 ┆ 0.312 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1130 │
│ microhecl ┆ pair-diag ┆ all-8.23        ┆ service ┆ 0.514 ┆ 0.514 ┆ 0.514 ┆ 0.514 ┆ 0.514 ┆  0.0 ┆ 0.514 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1130 │
│ shapleyiq ┆ pair-diag ┆ all-8.23        ┆ service ┆ 0.336 ┆ 0.533 ┆ 0.699 ┆ 0.845 ┆ 1.672 ┆  0.0 ┆ 0.486 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1130 │
│ microrca  ┆ pair-diag ┆ all-8.23        ┆ service ┆ 0.521 ┆ 0.599 ┆ 0.708 ┆ 0.815 ┆ 1.449 ┆  0.0 ┆ 0.605 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1130 │
│ microdig  ┆ pair-diag ┆ all-8.23        ┆ service ┆ 0.515 ┆ 0.743 ┆ 0.778 ┆ 1.167 ┆ 1.453 ┆  0.0 ┆ 0.633 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1129 │
└───────────┴───────────┴─────────────────┴─────────┴───────┴───────┴───────┴───────┴───────┴──────┴───────┴─────┴─────┴─────┴────────────┴────────────────┘
```
```sh
rca cross-dataset-metrics  -d pair-diag -dv all-may_anomaly-8.25 --tag 8.26run  -a microrank -a ton  -a microhecl -a shapleyiq  -a microrca -a microdig -l service
```
```
┌───────────┬───────────┬──────────────────────┬─────────┬───────┬───────┬───────┬───────┬───────┬────────┬───────┬─────┬─────┬─────┬────────────┬────────────────┐
│ algorithm ┆ dataset   ┆ dataset_version      ┆ level   ┆  top1 ┆  top3 ┆  top5 ┆  avg3 ┆  avg5 ┆   time ┆   mrr ┆ as1 ┆ as3 ┆ as5 ┆ efficiency ┆ datapack_count │
│ ---       ┆ ---       ┆ ---                  ┆ ---     ┆   --- ┆   --- ┆   --- ┆   --- ┆   --- ┆    --- ┆   --- ┆ --- ┆ --- ┆ --- ┆        --- ┆            --- │
│ str       ┆ str       ┆ str                  ┆ str     ┆   f64 ┆   f64 ┆   f64 ┆   f64 ┆   f64 ┆    f64 ┆   f64 ┆ f64 ┆ f64 ┆ f64 ┆        f64 ┆            i64 │
╞═══════════╪═══════════╪══════════════════════╪═════════╪═══════╪═══════╪═══════╪═══════╪═══════╪════════╪═══════╪═════╪═════╪═════╪════════════╪════════════════╡
│ microrank ┆ pair-diag ┆ all-may_anomaly-8.25 ┆ service ┆ 0.126 ┆  0.34 ┆ 0.466 ┆ 0.646 ┆ 1.237 ┆ 44.295 ┆ 0.298 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1700 │
│ ton       ┆ pair-diag ┆ all-may_anomaly-8.25 ┆ service ┆ 0.188 ┆ 0.322 ┆ 0.447 ┆ 0.573 ┆ 1.201 ┆ 43.566 ┆ 0.317 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1701 │
│ microhecl ┆ pair-diag ┆ all-may_anomaly-8.25 ┆ service ┆ 0.474 ┆ 0.474 ┆ 0.474 ┆ 0.474 ┆ 0.474 ┆ 42.776 ┆ 0.474 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1701 │
│ shapleyiq ┆ pair-diag ┆ all-may_anomaly-8.25 ┆ service ┆ 0.412 ┆  0.58 ┆   0.7 ┆ 0.849 ┆ 1.484 ┆ 48.711 ┆ 0.538 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1701 │
│ microrca  ┆ pair-diag ┆ all-may_anomaly-8.25 ┆ service ┆  0.48 ┆ 0.567 ┆ 0.675 ┆ 0.799 ┆ 1.439 ┆ 43.476 ┆ 0.563 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1701 │
│ microdig  ┆ pair-diag ┆ all-may_anomaly-8.25 ┆ service ┆ 0.475 ┆  0.68 ┆  0.72 ┆ 1.059 ┆ 1.382 ┆ 19.668 ┆ 0.586 ┆ 0.0 ┆ 0.0 ┆ 0.0 ┆        0.0 ┆           1698 │
└───────────┴───────────┴──────────────────────┴─────────┴───────┴───────┴───────┴───────┴───────┴────────┴───────┴─────┴─────┴─────┴────────────┴────────────────┘
```