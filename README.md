
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