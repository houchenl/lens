import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import RecentNotes from '../components/RecentNotes.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('RecentNotes', RecentNotes)
  },
} satisfies Theme
