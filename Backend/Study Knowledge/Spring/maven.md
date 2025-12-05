# Maven

**配置相关**

```xml
<!--

<dependency>
	<scope>test<scope> 表示只在test中使用该依赖
</dependency>

-->
```

```shell
mvn clean // 清理编译或打包后的项目结构，删除target文件夹
mvn compile // 编译
mvn test // 测试
mvn site // 生成一个项目依赖结构文件
mvn package // 打包项目，生成jar/war文件
mvn install // 打包上传到本地仓库
mvn deploy // 打包上传到私服仓库
```

Maven下载后在系统配置好环境变量（cmd中`mvn -v`返回版本信息即完成）

Maven中的conf/settings.xml配置以下内容：

```xml
<!-- 配置仓库地址；在mirrors里配置镜像地址；在profiles里配置jdk相关配置 -->

<localRepository>D:\maven-repo</localRepository>

<mirror>
    <id>alimaven</id>
    <mirrorOf>central</mirrorOf>
    <name>aliyun maven</name>
    <url>https://maven.aliyun.com/repository/central</url>
</mirror>

<profile>
    <id>jdk-21</id>
    <activation>
        <jdk>21</jdk>
    </activation>
    <properties>
        <maven.compiler.release>21</maven.compiler.release>
    </properties>
</profile>
```

