# static/ - the studio page

No build step. A few ES modules and one stylesheet, served by `server.py` at `/`
and `/static/...`.

| file | what |
|---|---|
| `index.html` | the page. An import map points `three` at the vendored bundle |
| `style.css` | panel and canvas layout |
| `app.js` | scene, sliders, servo readout |
| `urdf.js` | URDF -> three.js scene graph, posing it, and its outlines. No library |
| `ik.js` | drag a handle, solve the arm to it |
| `live.js` | the page's one WebSocket to the studio: the topics it watches come down as they change, what it says goes up; and the studio's alerts, as cards in a corner of every page |
| `physics.js` | the MuJoCo simulation: its frames down, commands up, over that socket |
| `three.module.min.js` | **vendor** three.js |
| `OrbitControls.js` | **vendor** three.js example camera control |
| `STLLoader.js` | **vendor** three.js example mesh loader |
| `train.html`, `train.js`, `train.css` | the training page: runs from the `train` topic, their points from `/train/series` |
| `uPlot.iife.min.js`, `uPlot.min.css` | **vendor** uPlot 1.6.32, the training page's charts. MIT, (c) 2022 Leon Sorokin |

## What the page is

A view of a simulation, not of a description. A slider asks the servos for an angle
and the stream says where the arm actually got to, which is not always the same
place; contact is MuJoCo's answer, against the real collision meshes. `physics.js`
opens that connection on load and throws if the server has no simulation - see its
header for the split between asking for a pose and placing one.

The desk and the board are drawn here but sized by `scene.xml`, which arrives with
the rest of `/physics`, so the board on screen is the board an arm collides with.

The badge beside the title says **sim** or **real**. With `--real` the model stands
where the real arms are - placed, not stepped - and the sliders are off, because the next reading would undo
whatever they asked for. Each joint also carries a third number - where the servo
says it is - and it is marked when it disagrees with the picture by more than a few
degrees.

## Vendor provenance

**three.js r160** (npm `three@0.160.1`), MIT License, Copyright 2010-2023 three.js
authors.

- `three.module.min.js` - `https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.min.js`
- `OrbitControls.js` - `https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/controls/OrbitControls.js`
- `STLLoader.js` - `https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/loaders/STLLoader.js`

All three are **unmodified**. The two example modules import `three` by bare name;
rather than edit them, `index.html` maps that name with an import map, so the files
stay byte-identical to what those URLs serve and their provenance stays checkable.

They are vendored rather than fetched at runtime because a rig without internet
still has to come up, and nothing here should depend on another host staying
reachable.
