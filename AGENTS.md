# lens 项目约定

`lens` 是基于 VitePress 的个人博客仓库，文章位于 `think/`、`system/`、`ai-infra/`。

## 文章写作

新增、改写、扩充或审阅文章前，先阅读并遵循 [文章写作规范](docs/writing-guide.md)。用户在当前任务中提出的具体要求优先。

撰写注意力机制系列时，同时阅读[系列推进路线](docs/attention-series-roadmap.md)，沿用已有编号与前置关系；当前安排是先完成 DeepSeek 注意力设计路线，再展开其他模型与线性混合机制。

写作的三个核心标准是：**语言专业、原理深入、深入浅出且循序渐进**。采用技术博客的阅读节奏与论文的论证结构，覆盖概述、问题背景、必要基础、详细原理、分析或验证、适用边界与结论；章节名称、数量和篇幅按主题组织。

## 项目实现

- 文章使用 Markdown，提供 `title`、`date`、`description` frontmatter。
- 数学公式使用已有的 VitePress MathJax 支持；保持公式、符号解释和适用条件对应。
- 栏目侧边栏、顶部栏目入口及首页近期笔记由 `.vitepress/utils/posts.ts`、`.vitepress/data/posts.data.ts` 自动生成。
- 写作规范等维护文档放在 `docs/`；`AGENTS.md`、`README.md` 和 `docs/` 已通过 `srcExclude` 排除在站点页面之外。
- 新增或实质修改文章后，运行 `npm run build`，检查公式、站内链接与文章收录；代码和数值示例按其内容做必要验证。说明实际验证范围。
- 构建产生的 `.vitepress/.temp/` 是临时产物，完成检查后清理；保留与本次任务有关的源文件改动。
