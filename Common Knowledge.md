### 1. Conda 命令

```bash
'''查看所有环境'''
conda env list

'''创建与删除环境'''
conda create --n <env_name> python=<version>
conda remove -n <env_name> --all

'''激活与退出环境'''
conda activate <env_name>
conda deactivate

'''克隆'''
conda create --name <new_env_name> --clone <old_env_name>

'''备份成.yml文件与恢复'''
conda env export > environment.yml
conda env create -f environment.yml

'''检查 conda 源'''
conda config --show-sources
conda config --show channels

'''修改 conda 源'''
vim ~/.condarc # 然后修改 channels

'''离线克隆环境'''
conda create -p /path/to/new_env --clone /path/to/old_env --offline --copy

'''离线打包克隆环境'''
# src server
pip install conda-pack
conda pack -p /path/to/envs/src_env -o src_env.tar.gz
scp src_env.tar.gz user@target_server:/path/to/target_dir
# target server
mkdir -p /path/to/envs/target_env
tar -xzf /path/to/target_dir/src_env.tar.gz -C /path/to/envs/target_env
/path/to/envs/target_env/bin/conda-unpack
```

### 2. 简单的数据处理
```bash
'''比较两文件之间的差异'''
diff -u file1.txt file2.txt # 统一格式
diff -y file1.txt file2.txt # 并排
```
