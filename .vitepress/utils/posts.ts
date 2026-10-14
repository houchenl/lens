import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { comparePosts } from './post-order.mjs'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')

export interface Section {
  dir: string
  text: string
}

export const SECTIONS: Section[] = [
  { dir: 'think', text: '思考' },
  { dir: 'system', text: '系统' },
  { dir: 'ai-infra', text: 'AI-Infra' },
]

export interface Post {
  /** 文章标题（frontmatter title，缺省取一级标题） */
  text: string
  /** 站内路径，如 /think/hello */
  link: string
  /** frontmatter 发布时间（含秒和时区，兼容旧日期），可为空 */
  date: string
  /** frontmatter description，可为空 */
  description: string
  /** 所属栏目目录名 */
  dir: string
  /** 文件修改时间（毫秒），无 date 时用于排序 */
  mtime: number
}

function parseFrontmatter(src: string): { title: string; date: string; description: string } {
  const m = src.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  const fm = m ? m[1] : ''
  const get = (key: string): string => {
    const r = fm.match(new RegExp(`^${key}:\\s*(.+?)\\s*$`, 'm'))
    return r ? r[1].replace(/^["']|["']$/g, '') : ''
  }
  const heading = src.match(/^#\s+(.+?)\s*$/m)?.[1] ?? ''
  return { title: get('title') || heading, date: get('date'), description: get('description') }
}

/** 扫描所有栏目目录，按发布时间倒序排列；无日期的旧页面放在末尾 */
export function getPosts(): Post[] {
  const posts: Post[] = []
  for (const { dir } of SECTIONS) {
    const absDir = path.join(ROOT, dir)
    if (!fs.existsSync(absDir)) continue
    for (const file of fs.readdirSync(absDir)) {
      if (!file.endsWith('.md')) continue
      const absFile = path.join(absDir, file)
      const { title, date, description } = parseFrontmatter(fs.readFileSync(absFile, 'utf-8'))
      posts.push({
        text: title,
        link: `/${dir}/${file.replace(/\.md$/, '')}`,
        date,
        description,
        dir,
        mtime: fs.statSync(absFile).mtimeMs,
      })
    }
  }
  return posts.sort(comparePosts)
}

/** 生成某个栏目的侧边栏分组 */
export function sidebarFor(dir: string, text: string) {
  const items = getPosts()
    .filter((p) => p.dir === dir)
    .map(({ text, link }) => ({ text, link }))
  return [{ text, items }]
}
