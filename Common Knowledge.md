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
```

