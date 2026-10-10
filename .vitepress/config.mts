import { defineConfig } from 'vitepress'
import { SECTIONS, getPosts, sidebarFor } from './utils/posts'

// 各栏目的导航入口指向该栏目最新的文章
const navSections = SECTIONS.map(({ dir, text }) => {
  const latest = getPosts().find((p) => p.dir === dir)
  return latest ? { text, link: latest.link } : null
})

export default defineConfig({
  lang: 'zh-CN',
  title: '后尘',
  description: '为学日益，为道日损',
  base: '/lens/',
  cleanUrls: true,
  lastUpdated: true,

  themeConfig: {
    nav: [{ text: '首页', link: '/' }, ...navSections.filter((s) => s !== null)],

    // 侧边栏由 .vitepress/utils/posts.ts 扫描目录自动生成
    sidebar: Object.fromEntries(
      SECTIONS.map(({ dir, text }) => [`/${dir}/`, sidebarFor(dir, text)]),
    ),

    socialLinks: [{ icon: 'github', link: 'https://github.com/houchenl/lens' }],

    outline: {
      label: '本页目录',
    },
    lastUpdatedText: '最近更新',
    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },
  },
})
