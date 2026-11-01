// @ts-check
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// The front page draws the robot with the studio's own modules, read from the
// package rather than copied, so the picture here is the one the studio draws.
// They import three by its bare name; this site's copy is the one they get.
const three = fileURLToPath(new URL('./node_modules/three/build/three.module.js', import.meta.url));

export default defineConfig({
	site: 'https://woolimi.github.io',
	base: '/pinkrab',
	vite: {
		resolve: { alias: [{ find: /^three$/, replacement: three }] },
		server: { fs: { allow: ['..'] } },
	},
	integrations: [
		starlight({
			title: 'Pinkrab',
			description:
				'벽에 부착하는 6-DOF 양팔 로봇과 UMI 그리퍼 Hermit: 조립, 시연 수집, policy 학습과 실행.',
			// Korean is the source; other languages are added beside it as locales.
			locales: { root: { label: '한국어', lang: 'ko' } },
			components: {
				PageTitle: './src/components/PageTitle.astro',
				Search: './src/components/Search.astro',
				Sidebar: './src/components/Sidebar.astro',
				SiteTitle: './src/components/SiteTitle.astro',
				ThemeProvider: './src/components/ThemeProvider.astro',
				ThemeSelect: './src/components/ThemeSelect.astro',
			},
			// The sidebar width the reader dragged to (Sidebar.astro), set before
			// the page paints so the layout does not jump.
			head: [
				{
					tag: 'script',
					content:
						"try{var w=localStorage.getItem('pk-sidebar-width');if(w)document.documentElement.style.setProperty('--sl-sidebar-width',Math.max(200,Math.min(+w,560,innerWidth/2))+'px')}catch(e){}",
				},
			],
			favicon: '/favicon.svg',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/woolimi/pinkrab' }],
			// Code and terminal blocks stay dark on the light pages.
			expressiveCode: {
				themes: ['github-dark-default'],
				// A window: rounded, lifted off the page, its title bar a shade
				// lighter than the code. The dots are coloured in pinkrab.css.
				styleOverrides: {
					borderRadius: '0.6rem',
					borderColor: '#30363d',
					codeBackground: '#0d1117',
					frames: {
						frameBoxShadowCssValue: '0 16px 24px -12px rgba(44, 26, 37, 0.5)',
						terminalBackground: '#0d1117',
						terminalTitlebarBackground: '#161b22',
						terminalTitlebarBorderBottomColor: '#30363d',
						terminalTitlebarForeground: '#8b949e',
						terminalTitlebarDotsOpacity: '1',
						editorTabBarBackground: '#161b22',
						editorActiveTabBackground: '#0d1117',
						editorActiveTabIndicatorTopColor: '#e2488e',
					},
				},
			},
			customCss: [
				'@fontsource/ibm-plex-sans/400.css',
				'@fontsource/ibm-plex-sans/600.css',
				'@fontsource/ibm-plex-sans/700.css',
				'@fontsource/ibm-plex-sans-kr/400.css',
				'@fontsource/ibm-plex-sans-kr/600.css',
				'@fontsource/ibm-plex-sans-kr/700.css',
				'@fontsource/jetbrains-mono/400.css',
				'@fontsource/outfit/700.css',
				'pretendard/dist/web/variable/pretendardvariable-dynamic-subset.css',
				'./src/styles/pinkrab.css',
			],
			sidebar: [
				{
					label: '시작',
					items: [
						{ label: '소개', slug: 'start/introduction' },
						{ label: '설치', slug: 'start/install' },
					],
				},
				{
					label: '하드웨어',
					items: [
						{ label: '부품 목록', slug: 'hardware/bom' },
						{ label: '부품 출력', slug: 'hardware/print' },
						{ label: 'Pinkrab 조립', slug: 'hardware/assemble-arm' },
						{ label: 'Hermit 조립', slug: 'hardware/assemble-hermit' },
						{ label: '카메라 교체', slug: 'hardware/cameras' },
					],
				},
				{
					label: '사용',
					items: [
						{ label: '준비', slug: 'use/setup' },
						{ label: '첫 구동', slug: 'use/first-run' },
						{ label: '시연 수집', slug: 'use/collect' },
						{ label: 'dataset 정리', slug: 'use/curate' },
						{ label: '학습', slug: 'use/train' },
						{ label: '실행', slug: 'use/run' },
					],
				},
				{
					label: '심화',
					items: [
						{ label: 'Hermit 과 teleoperation', slug: 'deep/teleoperation' },
						{ label: '시연 데이터와 학습', slug: 'deep/data' },
						{ label: '교정 데이터', slug: 'deep/corrections' },
						{ label: 'dataset 변환', slug: 'deep/convert' },
						{
							label: 'policy',
							items: [
								{ label: '개요', slug: 'deep/policies' },
								{ label: 'ACT', slug: 'deep/policies/act' },
							],
						},
					],
				},
			],
		}),
	],
});
