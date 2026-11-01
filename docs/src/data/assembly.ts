// The assembly steps the hardware pages walk through, one list per assembly,
// from assembly.json: each step's words and the parts it adds. A part is named
// by where it sits:
//   arm     `<link>:<file>` - the link without its side, the mesh without
//           its side and extension (`shoulder_link:shoulder` is the left arm's
//           shoulder_left.stl on left_shoulder_link)
//   hermit  one of the Hermit's own parts, as assembly.js lays them out
// A step with no parts is one done by hand on the whole: the cables.
//
// The screws are found in the meshes by scripts/assembly.py and drawn going
// in: the base's bolts through the board; a servo's - into its holder, the
// top horn's own, and every one through a part into its horns, where a part
// has a hole over the servo's - named as on the SO-ARM101 the arm is built
// from (M2×6 into the case, M3×6 into the horns) unless a step names its horn
// screw; and those through a counterbored hole of one printed part into
// another, sized by the hole, with the nuts an M3 bolt takes put in at the
// step that names them. Only the camera module's own screws, which hold
// no printed part, are written down with their step in assembly.json.

import data from './assembly.json';
import found from '../../public/robot/assembly.json';
import { isEn } from '../i18n';

export interface Screw {
	size: string;
	count: number;
	/** What it holds, a few words. */
	what: string;
	/** The English of `what`, for a screw written in a step. */
	whatEn?: string;
}

export interface Step {
	title: string;
	text: string;
	parts: string[];
	screws: Screw[];
}

type Kind = 'arm' | 'hermit';
interface Written {
	title: string;
	text: string;
	parts: string[];
	screws?: Screw[];
	/** The English title and text. */
	en: { title: string; text: string };
	horn?: string;
	/** The part whose bolts take the nuts this step puts in. */
	nuts?: string;
	/**
	 * A servo board screwed onto the step's part before the part goes on: the
	 * edge its servo sockets face, and per side where its USB-C is (see
	 * scripts/assembly.py).
	 */
	board?: {
		servo: { axis: number[]; of?: string };
		xiao?: boolean;
		left: { usb: 'in' | 'out' | 'back' };
		right: { usb: 'in' | 'out' | 'back' };
	};
}
interface Found {
	step: number;
	kind: 'centre' | 'horn' | 'mount' | 'bolt' | 'join' | 'nut';
	servo: string;
	part: string;
	/** A join's size, read off its hole. */
	size?: string;
}

const SIZES = { centre: 'M3×6', horn: 'M3×6', mount: 'M2×6', bolt: 'M4×20 접시머리 + M4 너트' };
const SIZES_EN = { ...SIZES, bolt: 'M4×20 countersunk + M4 nut' };

/** The steps of an assembly, in the page's language. */
export function stepsOf(kind: Kind, lang: string): Step[] {
	const en = isEn(lang);
	const written = data[kind] as Written[];
	// A part by its mesh's name: `upper_arm_link:upper_arm` is the upper arm.
	const nameOf = (key: string) => key.split(':').pop()!.replace(/_/g, ' ');
	// The left side's: the right is its mirror, screw for screw.
	const screws = (found[kind].left.screws as Found[]);
	return written.map((step, i) => {
		const here = new Map<string, Screw>();
		for (const s of screws.filter((s) => s.step === i)) {
			const size = s.kind === 'join' || s.kind === 'nut' ? (en ? s.size!.replace('육각 너트', 'hex nut').replace('접시머리', 'countersunk') : s.size!)
				: s.kind === 'horn' ? (step.horn ?? SIZES.horn) : (en ? SIZES_EN : SIZES)[s.kind];
			const what = s.kind === 'nut' ? (en ? `${nameOf(s.part)} nut slot` : `${nameOf(s.part)} 너트 자리`)
				: s.kind === 'join' ? (en ? `${nameOf(s.part)} fixing` : `${nameOf(s.part)} 고정`)
				: s.kind === 'bolt' ? (en ? 'Base → pegboard' : '베이스 → 페그보드')
				: s.kind === 'centre' ? (en ? 'Top horn, centre' : '윗면 혼 가운데')
				: s.kind === 'horn' ? (en ? `${nameOf(s.part)} → horn` : `${nameOf(s.part)} → 혼`)
				: (en ? `Servo → ${nameOf(s.part)}` : `서보 → ${nameOf(s.part)}`);
			const key = `${size}|${what}`;
			const seen = here.get(key);
			if (seen) seen.count += 1;
			else here.set(key, { size, count: 1, what });
		}
		const own = (step.screws ?? []).map((s) => ({ ...s, what: en ? s.whatEn ?? s.what : s.what }));
		return {
			title: en ? step.en.title : step.title,
			text: en ? step.en.text : step.text,
			parts: step.parts,
			screws: [...here.values(), ...own],
		};
	});
}

/** Every screw of an assembly, by size, for the parts list. */
export function screwTotals(list: Step[]): { size: string; count: number }[] {
	const totals = new Map<string, number>();
	for (const step of list) for (const s of step.screws) totals.set(s.size, (totals.get(s.size) ?? 0) + s.count);
	return [...totals].map(([size, count]) => ({ size, count }));
}
