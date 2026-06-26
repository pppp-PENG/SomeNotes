### 知识点

##### MLA (Multi-head Latent Attention)
---

| 方法      | K/V 设计                  | KV Cache 大小 | 表达能力 | 主要特点                              |
| ------- | ----------------------- | ----------: | ---- | --------------------------------- |
| **MHA** | 每个 Query head 都有独立 K/V  |          最大 | 强    | 标准多头注意力，效果好但推理 cache 显存大          |
| **MQA** | 所有 Query heads 共享一组 K/V |          最小 | 较弱   | 极大减少 KV Cache，但可能损失多头表达能力         |
| **GQA** | 多个 Query heads 共享一组 K/V |          中等 | 中等   | MHA 与 MQA 的折中，常用于高效 LLM           |
| **MLA** | K/V 先压缩为 latent，再参与注意力  |          很小 | 强    | 不简单共享 K/V，而是缓存低维 latent，兼顾效率和表达能力 |

---
