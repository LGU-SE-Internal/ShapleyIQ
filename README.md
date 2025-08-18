
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

┌───────────────────┬───────────┬───────┬───────┬─────────────────────┬──────────┬────────────┬────────────┬────────────┬──────────┬──────────┬─────────┐
│ dataset           ┆ algorithm ┆ total ┆ error ┆ runtime.seconds:avg ┆      MRR ┆ AC@1.count ┆ AC@3.count ┆ AC@5.count ┆     AC@1 ┆     AC@3 ┆    AC@5 │
│ ---               ┆ ---       ┆   --- ┆   --- ┆                 --- ┆      --- ┆        --- ┆        --- ┆        --- ┆      --- ┆      --- ┆     --- │
│ str               ┆ str       ┆   u32 ┆   u32 ┆                 f64 ┆      f64 ┆        f64 ┆        f64 ┆        f64 ┆      f64 ┆      f64 ┆     f64 │
╞═══════════════════╪═══════════╪═══════╪═══════╪═════════════════════╪══════════╪════════════╪════════════╪════════════╪══════════╪══════════╪═════════╡
│ rcabench_filtered ┆ shapleyiq ┆   438 ┆     0 ┆           29.226472 ┆ 0.334031 ┆       81.0 ┆      146.0 ┆      232.0 ┆ 0.184932 ┆ 0.333333 ┆ 0.52968 │
└───────────────────┴───────────┴───────┴───────┴─────────────────────┴──────────┴────────────┴────────────┴────────────┴──────────┴──────────┴─────────┘

我们的系统的 trace 加了 loadgenerator 和 caddy，这两个入口和前端导致整个他俩基本上都排前二