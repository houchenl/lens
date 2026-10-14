/** @param {{ date: string, link: string }} post */
function publicationTimestamp(post) {
  if (!post.date) return -Infinity

  // 兼容旧的日期写法，按站点时区（北京时间）当天零点解释。
  const date = /^\d{4}-\d{2}-\d{2}$/.test(post.date)
    ? `${post.date}T00:00:00+08:00`
    : post.date
  const timestamp = Date.parse(date)
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$/.test(date)
    || !Number.isFinite(timestamp)) {
    throw new Error(`${post.link}: date 请使用带时区的秒级时间，如 2026-10-14T17:00:50+08:00`)
  }
  return timestamp
}

/**
 * 发布时间倒序；同一秒使用路径的自然倒序，保证不同机器上结果一致。
 * 仅无发布时间的旧页面继续使用文件修改时间，并排在有日期的文章之后。
 * @param {{ date: string, link: string, mtime: number }} a
 * @param {{ date: string, link: string, mtime: number }} b
 */
export function comparePosts(a, b) {
  const aTime = publicationTimestamp(a)
  const bTime = publicationTimestamp(b)
  if (aTime !== bTime) return aTime > bTime ? -1 : 1
  if (!a.date && !b.date && a.mtime !== b.mtime) return b.mtime - a.mtime
  return b.link.localeCompare(a.link, 'en', { numeric: true })
}
