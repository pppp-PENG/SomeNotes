### 1. MySQL基础

**基础架构**

MySQL 分为两层：

- Server 层——建立连接、分析和执行 SQL
- 数据库引擎——数据的存储和提取

**MySQL 执行流程**

1. 连接器：MySQL 与客户端基于 TCP 传输，有最大空闲时间和最大连接数量，超过即断开连接或拒绝连接
2. 查缓存：在 Server层，MySQL 8.0 已移除
3. 解析 SQL：检查语法、构建语法树
4. 执行 SQL：
    - prepare（预处理）检查 SQL 查询语句中的表或者字段是否存在
    - optimize（优化）确定 SQL 查询语句的执行方案
    - execute（执行）根据执行计划执行 SQL 查询语句，从存储引擎读取记录，返回给客户端

**存储结构**

`mysql/db.op` 存储默认字符集和字符校验规则

`mysql/xxx.frm` 存储某一表的表结构

`mysql/xxx.ibd` 存储某一表的表数据

`ibd` （表空间文件）结构：

- 行（row）：记录的存放单位
- 页（page）：IO 读取单位，==**一般为16KB**==（一次读一页，减少IO次数）
- 区（extent）：为了让相邻的页在同一区内，减少随机 IO，一般为1MB
- 段（segment）：索引段、数据段、回滚段

COMPACT 行格式：

![image-20241116132945231](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20241116132945231.png)

**行溢出**

MySQL 中要保持 ==**$字段长度 + 变长字段字节数列表长度 + null 值表长度 <= 64KB$**==，但一页的大小为 16KB

当字段长度大于页大小时，发生行溢出，真实数据写在溢出页中，原本记录数据的地方存储指向真实数据的指针

### 2. 索引

**索引分类**

1. 数据结构：B+ Tree（InnoDB 引擎）、Hash、Full-Text
2. 物理存储：聚簇索引（主键索引）、二级索引
3. 字段特性：主键索引、唯一索引、普通索引、前缀索引
4. 字段个数：单列索引、联合索引

**聚簇索引**

1. 若有设置主键，默认将其作为聚簇索引的索引键
2. 若无设置主键，选择第一个不包含 NULL 的唯一列作为聚簇索引的索引健
3. 在上面两个都没有的情况下，InnoDB 将自动生成一个隐式自增 id 列作为聚簇索引的索引键

**常见索引失效情况**

- “左原则”
    - 左模糊匹配
    - 联合索引非最左匹配
- 转换类（等号左边不是索引时）
    - 使用函数：如匹配 `length(column1) = 10` 时，若无长度索引，则会进行全表扫描
    - 表达式计算：如匹配 `id + 1 = 10` 时，MySQL 并不会优化，而是会像使用函数一样走全表扫描
    - 隐式类型转换：MySQL 匹配字符串与数字比较时，会优先把字符串转化为数字，故若表中为 `varchar` 类型，而查询语句使用数字，那么就会走全表扫描
- or 的使用：or 只会在左右两个条件都是索引时才会走索引扫描然后取并集，否则直接走全表扫描

**常见索引优化方法**

- 前缀索引优化：用前缀代替整个字段，减少索引项大小，索引页可以存储更多索引
- 覆盖索引优化：联合索引中包括了需要查找的字段，避免了回表操作
- 主键索引自增：自增的主键能有更快的插入速度（顺序插入），避免了页分裂情况
- NOT NULL 设置：减少行格式中的  NULL 值列表

**执行计划**

type 字段：

- All：全表扫描
- index：全索引扫描
- range：索引范围扫描（如`BETWEEN`、`>`）
- ref：使用非唯一索引匹配多行（如`=`）
- eq_ref：使用唯一索引匹配单行（多表时出现，如`join`）
- const：通过主键或唯一索引直接定位单行

**count() 函数**

- count(1) / count(*) / count(主键)：使用 key_len 最短的二级索引进行全索引扫描（如没有二级索引则使用主键索引）
- count(普通字段)：使用全表扫描

### 3. 事务

**事务特性**

**==原子性、一致性、隔离性、持久性==**

**并行事务**

- 脏读：事务读到了==**另一个事务修改但未提交**==的数据
- 不可重复读：一个事务两次读取同一个记录内容不一致
- 幻读：一个事务两次读取的结果集不一致

**SQL 事务隔离级别**

- 读未提交：事务未提交时即可被其他事务看到
- 读提交：事务提交后才可被其他事务看到【避免脏读】
- 可重复读：一个事务看到的数据维持在启动时的数据状态【避免不可重复读】
- 串行化：加上读写锁，避免读写冲突【避免幻读】

**MySQL InnoDB 引擎默认隔离级别**

MVCC 保证了可重复读，并**==解决快照读的幻读，next-key lock （记录锁 + 间隙锁）解决当前读的幻读==**（但未完全解决幻读）

可能的幻读现象：快照读后当前读，两次读可能不一致，因为当前读并不会根据 Read View 判断记录版本，所以最好在事务开始时直接先当前读（加锁）

**MVCC**

在每个事务创建时同步生成一个 ==**Read View**==，包含本事务的编号 trx_id（按时间递增），已创建未提交的事务 id 集合 m_ids，m_ids 的最大最小值 min_trx_id、max_trx_id

在每行记录有两个隐藏列，分别存储更新当前记录的事务编号 trx_id（版本号）和 undo log 指针（回滚指针，指向上个版本的记录）

1. 判断读取记录的 trx_id 是否小于本事务 m_ids 最小值（判断记录版本号是否是本事务创建之前）
2. 若是，则读该数据；若不是，则依据 undo log 寻址找到上一个版本，返回第 1 步

### 4. 锁

**全局锁**

锁住整个数据库，但会使整个数据库处于只读状态

常用于==**不支持可重复读的数据库引擎**==的数据库备份

**表级锁**

- 表锁：对一张表加锁
- 元数据锁（MDL）：CRUD 操作获取读锁，表结构更改操作获取写锁
- 意向锁：快速判断表里是否有记录被加锁（行级需要遍历，表级意向锁不需要）
- AUTO-INC 锁：插入不含主键的数据时，赋予自增 id 的锁

**行级锁**

- Record Lock（记录锁）分为 S 锁和 X 锁
- Gap Lock（间隙锁）是前开后开区间，其意义只是阻止区间被插入
- Next-Key Lock（上两种的组合锁）是前开后闭区间，根据情况退化成前两种锁

**如何打破死锁**

死锁四个必要条件：互斥、占有且等待、不可抢夺、循环等待

打破循环等待的方法：

- 设置事务等待锁的超时时间——事务等待时间超过一定值时回滚，释放锁
- 开启死锁检测——检测到死锁会回滚其中一个事务

### 5. 日志

### 6. 内存













MySQL 语句

```shell
''' 连接 MySQL '''
mysql -u root -p

''' 查看 MySQL 服务当前连接情况 '''
show processlist
```

```sql
-- 基础语法
create database database_name
create table tab_name (
	col_1 type_1 [not null] [primary key],
	col_2 type_2 [not null] [primary key],
    …
)
drop database database_name
drop table tab_name
select * from tab_name where condition_
select * from tab_name where condition_ like '%ABC_'
select col_1, count(col_2) from tab_name group by col_1 having count(col_2) > 2
insert into tab_name(col_1,col_2) values(value_1,value_2)
delete from tab_name where condition_
update tab_name set col_1=balue_1 where condition_

-- 创建索引
create table table_1(
    column_1 datatype,
    column_2 datatype,
    ...
    index index_name(column_1, column_2, ...)
)
create index index_name on table_1(column_1, column_2, ...)

-- 全局锁及其释放
flush tables with read lock
unlock tables

-- 添加读/写表锁
lock table table_1 read
lock table table_1 write

-- S锁与X锁
select * from table_1 where condition_1 lock in share mode
select * from table_1 where condition_1 for update
```

