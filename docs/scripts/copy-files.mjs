// Files the site serves but does not own, copied out of the packages that do at
// every dev and build: the meshes the robot is drawn from, into public/robot
// where the exported description points, and the print projects and the lens
// board the hardware pages offer for download, into public/print.
import { cpSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const site = fileURLToPath(new URL('../public/', import.meta.url));
const copies = [
	[`${repo}src/pinkrab_description/meshes`, `${site}robot/meshes`, ['.stl']],
	[`${repo}src/pinkrab_umi/meshes`, `${site}robot/hermit/meshes`, ['.stl']],
	[`${repo}src/pinkrab_description/print`, `${site}print`, ['.3mf']],
	[`${repo}src/pinkrab_umi/print`, `${site}print`, ['.3mf', 'lens_charuco_a4.pdf', 'lens_charuco_a4.svg']],
];

for (const to of new Set(copies.map(([, to]) => to))) {
	rmSync(to, { recursive: true, force: true });
	mkdirSync(to, { recursive: true });
}
for (const [from, to, kinds] of copies) {
	for (const name of readdirSync(from).filter((n) => kinds.some((k) => n.endsWith(k)))) {
		cpSync(`${from}/${name}`, `${to}/${name}`);
	}
}
