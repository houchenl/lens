import assert from 'node:assert/strict'
import { test } from 'node:test'
import { comparePosts } from './post-order.mjs'

const post = (date, link = '/ai-infra/article', mtime = 0) => ({ date, link, mtime })

test('同一天按时分秒倒序，文件名与修改时间不影响发布时间', () => {
  const older = post('2026-10-14T17:00:49+08:00', '/z', 100)
  const newer = post('2026-10-14T17:00:50+08:00', '/a', 0)
  assert.deepEqual([older, newer].sort(comparePosts), [newer, older])
})

test('不同时区按实际时间比较，不能比较时间字符串', () => {
  const older = post('2026-10-14T18:00:00+09:00')
  const newer = post('2026-10-14T09:00:01Z')
  assert.deepEqual([older, newer].sort(comparePosts), [newer, older])
})

test('旧日期按北京时间零点解释，同日有时间的新文章排在前面', () => {
  const legacy = post('2026-10-14')
  const newer = post('2026-10-14T00:00:01+08:00')
  assert.deepEqual([legacy, newer].sort(comparePosts), [newer, legacy])
  assert.equal(comparePosts(legacy, post('2026-10-13T16:00:00Z')), 0)
})

test('同一秒使用文件路径的自然倒序，重建后顺序不受 mtime 影响', () => {
  const date = '2026-10-14T16:10:25+08:00'
  const posts = [post(date, '/attention-4', 300), post(date, '/attention-10', 100), post(date, '/attention-5', 200)]
  assert.deepEqual(posts.sort(comparePosts).map(p => p.link), ['/attention-10', '/attention-5', '/attention-4'])
})

test('无日期页面排在已发布文章之后，并兼容修改时间倒序', () => {
  const dated = post('2026-10-10T00:00:00+08:00')
  const old = post('', '/old', 100)
  const recent = post('', '/recent', Date.now())
  assert.deepEqual([old, recent, dated].sort(comparePosts), [dated, recent, old])
})

test('拒绝无时区的时间和错误日期，避免构建环境改变文章顺序', () => {
  for (const date of ['2026-10-14T17:00:50', 'not-a-date', '2026-13-14T17:00:50+08:00']) {
    assert.throws(() => comparePosts(post(date), post('2026-10-14')), /date 请使用带时区的秒级时间/)
  }
})
