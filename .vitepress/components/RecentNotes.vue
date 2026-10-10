<script setup lang="ts">
import { withBase } from 'vitepress'
import { data } from '../data/posts.data'

const RECENT_COUNT = 5
</script>

<template>
  <div class="recent">
    <ul>
      <li v-for="post in data.slice(0, RECENT_COUNT)" :key="post.link">
        <a :href="withBase(post.link)">{{ post.text }}</a>
        <span v-if="post.description" class="desc">{{ post.description }}</span>
      </li>
    </ul>
    <p v-if="data.length === 0">暂无笔记。</p>
  </div>
</template>
