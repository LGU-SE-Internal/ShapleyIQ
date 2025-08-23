
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
docker build -t 10.10.10.240/library/rca-algo-microdig:latest .
docker push 10.10.10.240/library/rca-algo-microdig:latest
rca upload-algorithm-harbor ./
```
```sh
# batch test
sudo -E .venv/bin/python run.py batch-test --label 8.15microdig
```