### **RL4RS方案调研报告**

##### 一、方案核心思路回顾

​	原方案拟利用 RL “优化未来期望收益最大化”的特性，解决传统生成式推荐系统仅关注短期（ t+1 步）预测的局限性，并进一步探讨 RL 在冷启动及新物品适应能力上是否优于监督学习。

##### 二、关键调研过程（其中较相关的论文）

##### **1. 通过 RL 解决传统推荐的短期预测局限**

​	前人已在该领域有较充分探索：例如，https://arxiv.org/pdf/1902.05570《Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems》论文通过强化学习框架显式建模用户长期交互价值，而非仅优化单步（t+1）推荐结果。该研究证实，相比传统监督学习（如协同过滤、CTR模型），RL 能通过序列决策机制更精准地捕捉用户长期兴趣，从而提升整体推荐收益。

​	思考：前人确实已经在RL上有了一定的探索了，我们最初的想法已经算是最基础的idea了，因此若走这条路，需要寻找gap！

##### 2. **目前前沿研究中 RL 的应用方向**

​	https://ojs.aaai.org/index.php/AAAI/article/view/3815《Hierarchical Reinforcement Learning for Course Recommendation in MOOCs》论文针对推荐系统中常见的标注噪声问题（如用户虚假点击、数据缺失），提出分层强化学习框架。该模型无需依赖专家标注，而是通过分层策略自动筛选有效交互信号并抑制噪声干扰，在保证推荐稳定性的同时提升了模型鲁棒性。

​	https://arxiv.org/pdf/2503.24289《Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning》论文将 RL 作为大语言模型（LLM）与推荐系统的连接桥梁：通过RL机制将用户实时反馈（如点击、停留）转化为 LLM 的微调信号，使大模型能动态适配用户偏好变化。

​	以上论文和近几年其他 RL4RS 的工作都表明，RL 在推荐系统中的基础应用已被广泛验证，而我们最初提出的“通过 RL 解决短期预测局限”属于该方向的基础尝试，当前需寻找更具体的gap。

##### 3. **离线测试的可靠性**

​	https://arxiv.org/pdf/2309.12645《KuaiSim: A Comprehensive Simulator for Recommender Systems》与 https://arxiv.org/pdf/1909.04847《RECSIM: A Configurable Simulation Platform for Recommender Systems》分别提出了 KuaiSim 和 RecSim 两套推荐系统专用模拟器。这两类工具通过构建虚拟用户群体（模拟点击、浏览、购买等行为）、设定动态偏好参数（如兴趣漂移、季节性变化），可在无真实数据的情况下复现部分推荐场景。

​	此外，https://github.com/fuxiAIlab/RL4RS 也提供了开源数据集 RL4RS。

##### 三、小结

​	经调研后，目前的初步想法有：1. 以优化 RL 在推荐系统表现力为目标（如结构优化、多模型优化等）；2. 以 RL 作为桥梁对其他已有工作进行优化（如大模型推荐系统 + RL、召回排序系统 + RL等）。但具体目标仍需调研现有文献寻找 gap。

​	且对于模拟器构造的环境，难以保证其能完全复现真实用户的动态偏好与复杂交互逻辑。现有文献中，基于模拟器的训练测试方法较少被主流研究采用（多数可靠成果依赖在线A/B测试），有可能是因为离线结果的说服力有限，难以支撑方案的可靠性验证。