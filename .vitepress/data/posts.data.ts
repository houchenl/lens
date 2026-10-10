import { getPosts, type Post } from '../utils/posts'

export declare const data: Post[]

export default {
  // 文章增删改时重新加载（dev 模式热更新）
  watch: ['think/*.md', 'system/*.md', 'ai-infra/*.md'],
  load() {
    // 栏目的 hello 开篇页不算「笔记」，不进入近期列表
    return getPosts().filter((p) => !p.link.endsWith('/hello'))
  },
}
