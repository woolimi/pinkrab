# Pinkrab documentation

The documentation site, built with [Starlight](https://starlight.astro.build) and published to GitHub Pages by `.github/workflows/docs.yml`.

```bash
npm install
npm run dev      # http://localhost:4321/pinkrab/
npm run build    # into dist/
```

The front page draws the robot with the studio's own modules (`src/pinkrab_studio/pinkrab_studio/static`). What the studio's server answers them with is written into `public/robot/` by

```bash
pinkrab-studio --export docs/public/robot
```

and has to be run again when the description, the ready pose or Hermit changes. The meshes, and the print projects the printing page offers for download, are copied from the packages at every `dev` and `build` (`scripts/copy-files.mjs`).

The assembly pages' steps are in `src/data/assembly.json`. Which way each step's parts go on and where every servo's screws go are worked out from the meshes into `public/robot/assembly.json` by

```bash
python docs/scripts/assembly.py
```

run after the export, and again when the steps change (it takes a few minutes).
