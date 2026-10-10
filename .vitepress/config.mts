import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: '后尘',
  description: '为学日益，为道日损',
  base: '/lens/',
  cleanUrls: true,
  lastUpdated: true,

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '思考', link: '/think/dao-fa-shi-shu-qi' },
      { text: '系统', link: '/system/hello' },
      { text: '技术', link: '/tech/hello' },
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
      '/system/': [
        {
          text: '系统',
          items: [{ text: '你好，系统', link: '/system/hello' }],
        },
      ],
      '/tech/': [
        {
          text: '技术',
          items: [{ text: '你好，技术', link: '/tech/hello' }],
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
