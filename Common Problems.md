### IDE中文乱码情况

问题背景：使用cursor写java时，System输出中文时出现乱码

原因分析：系统未设置非Unicode程序的相关兼容

解决方法：进入 "设置" -> "时间和语言" -> "语言和区域" -> "管理语言设置" -> "更改系统区域设置"，勾选 "使用 Unicode UTF-8 提供全球语言支持"

### 下载docker镜像失败

问题背景：下载docker镜像（如MySQL时出现网络错误）

原因分析：无法进行外网连接

解决方法：使用国内镜像（目前阿里云等都不可用）

方案一：

```shell
vim /etc/docker/daemon.json
```

然后在文本中添加

```json
{
	"registry-mirrors" : [
		"https://docker.1ms.run"
	]
}
```

方案二：（临时）

```shell
docker pull docker.1ms.run/mysql
```



