# lens · 后尘

- 站点：基于 VitePress，`npm install && npm run dev` 本地预览，`npm run build` 构建。

## 文章写作

新增或修改文章时，遵循[文章写作规范](docs/writing-guide.md)：语言专业、原理深入、深入浅出且循序渐进，融合技术博客的表达与论文的篇章逻辑。

[AGENTS.md](AGENTS.md) 为协作工具提供项目入口，引用同一份规范。

[注意力机制系列推进路线](docs/attention-series-roadmap.md)记录后续篇目、前置关系与资料入口：先完成 DeepSeek 主线，再展开 GLM、Kimi、MiniMax 的不同注意力设计。

## 部署

push 到 `main` 分支后 GitHub Actions 自动发布到 GitHub Pages（仓库 Settings → Pages → Source 选 GitHub Actions）。线上地址：https://houchenl.github.io/lens/
