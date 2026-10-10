import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: '思考手记',
  description: '思考 · 系统 · 技术',
  base: '/lens/',
  cleanUrls: true,
  lastUpdated: true,

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '思考', link: '/think/dao-fa-shi-shu-qi' },
    ],

    sidebar: {
      '/think/': [
        {
          text: '思考',
          items: [
            { text: '道法势术器：一个理解世界的五层框架', link: '/think/dao-fa-shi-shu-qi' },
            { text: '你好，手记', link: '/think/hello' },
          ],
        },
      ],
    },

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
