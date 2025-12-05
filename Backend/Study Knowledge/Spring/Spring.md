# Spring

### Spring特点

1. Core Container 核心容器
2. AOP&Aspect 面向切面编程
3. TX 声明式事务管理
4. Spring MVC 提供面向Web应用程序的集成功能

### 容器实例

ClassPathXmlApplicationContext

AnnotationConfigApplicationContext

WebApplicationContext

### 容器配置方式

XML配置

注解

Java配置类

### IoC/DI

IoC（Inversion of Control）控制反转：对象的创建和管理由容器负责，而非程序自己

DI（Dependency Injection）依赖注入：组件间传递依赖过程在容器中处理，不必在代码中硬编写依赖关系，注入方式有构造函数注入、setter方法注入、接口注入

### XML注入

```java
public class People {
    private String name;
    private PeopleSex sex;
    
    // 构造方法
    public People(String name, PeopleSex sex) {
        this.name = name;
        this.sex = sex;
    }
    
    // setter方法
    public setPeopleName(String name) {
        this.name = name;
    }
    public setPeopleSex(PeopleSex sex) {
        this.sex = sex;
    }
    
    // 设置初始化和销毁方法（创建IoC时调用初始化方法，close时调用销毁方法）
    public void init() {System.out.println("Init successful!");}
    public void clean() {System.out.println("Clean successful!");} // 需要容器主动close
}
```

```xml
<!-- 直接注入 -->
<bean id="对象1" class="com.example.People"/>
<bean id="sexObject" class="com.example.PeopleSex">

<!-- 构造方法注入，写好构造方法参数名及对应值 -->
<bean id="对象2" class="com.example.People">
    <constructor-arg name="name" value="小明"/>
    <constructor-arg name="sex" ref="sexObject"/>
</bean>

<!-- setter方法注入 -->
<bean id="对象3" class="com.example.People">
    <property name="name" value="小明"/>
    <property name="sex" ref="sexObject"/>
</bean>
    
<!-- 设置初始化和销毁方法 -->
<bean id="对象4" class="com.example.People" init-method="init" destroy-method="clean"/>
```

### IoC创建与使用

```java
// 创建IoC容器
ApplicationContext applicationcontext = new ClassPathXmlApplicationContext("xxx.xml");

// 使用getBean方法获取Bean，参数1为id，参数2为类名（如果无多个同类型Bean可以省略参数1）
People people = applicationContext.getBean("对象2", People.class);
```



