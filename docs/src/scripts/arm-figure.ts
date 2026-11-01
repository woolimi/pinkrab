// Starts a figure's picture once the figure comes near the screen: the arm's
// description and meshes are fetched and the stage is built only then.
const robot = `${import.meta.env.BASE_URL.replace(/\/$/, '')}/robot`;
export const SOURCES = {
	physics: `${robot}/physics.json`,
	model: `${robot}/model.urdf`,
	ready: `${robot}/ready.json`,
	meshes: `${robot}/meshes`,
	workspace: `${robot}/workspace.json`,
	hermit: `${robot}/hermit.json`,
	hermitMeshes: `${robot}/hermit/meshes`,
	assembly: `${robot}/assembly.json`,
};

export function whenNear(figure: HTMLElement, start: () => Promise<unknown>) {
	const watch = new IntersectionObserver(
		([entry]) => {
			if (!entry.isIntersecting) return;
			watch.disconnect();
			// Without the picture, what the figure says in words is still there.
			start().catch(() => figure.classList.add('no-robot'));
		},
		{ rootMargin: '400px 0px' },
	);
	watch.observe(figure);
}
