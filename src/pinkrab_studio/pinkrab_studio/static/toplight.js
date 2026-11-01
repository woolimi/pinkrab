// The top camera's light, set while it runs: two sliders - the exposure, and
// the bright end the gain follows - and what the camera holds now beside them.
// The exposure is the hand's blur budget as much as the picture's brightness:
// longer is brighter and smears a moving marker more. The target only moves
// the gain, which brightens without smearing and adds noise; once the gain is
// at its top, only the exposure brightens further. A camera with no gain to
// set has the exposure slider alone.

const SEND_MS = 150;

/**
 * Light controls in `host`. `show(light)` is called with the stats' head_light
 * (null while the camera is not open); a slider being moved is left alone.
 */
export function topLight(host) {
  host.classList.add('top-light');
  host.innerHTML = `
    <label>exposure <input type="range" class="exposure" step="0.5"><output class="exposure-out"></output></label>
    <label>brightness <input type="range" class="target" min="40" max="240" step="5"><output class="target-out"></output></label>
    <span class="held"></span>`;
  const exposure = host.querySelector('.exposure');
  const target = host.querySelector('.target');
  const exposureOut = host.querySelector('.exposure-out');
  const targetOut = host.querySelector('.target-out');
  const held = host.querySelector('.held');
  let timer = null;
  let touched = 0;
  let gainControl = true;

  const say = () => {
    exposureOut.textContent = `${Number(exposure.value).toFixed(1)} ms`;
    targetOut.textContent = target.value;
  };
  const send = () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      try {
        const response = await fetch('/rig/live/head/light', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(gainControl
            ? { exposure_ms: Number(exposure.value), target: Number(target.value) }
            : { exposure_ms: Number(exposure.value) }),
        });
        const answer = await response.json();
        held.textContent = response.ok ? '' : (answer.error ?? response.statusText);
      } catch (err) {
        held.textContent = String(err);
      }
    }, SEND_MS);
  };
  for (const slider of [exposure, target]) {
    slider.addEventListener('input', () => { touched = performance.now(); say(); send(); });
  }

  return function show(light) {
    host.hidden = !light;
    if (!light) return;
    exposure.min = light.exposure_min_ms;
    exposure.max = light.exposure_max_ms;
    gainControl = light.gain_control !== false;
    target.disabled = !gainControl;
    // What the camera holds, unless a slider was just moved: the answer to
    // that move is a frame or two behind it.
    if (performance.now() - touched > 1500) {
      exposure.value = light.exposure_ms;
      if (gainControl) target.value = light.target;
      say();
    }
    if (!gainControl) {
      targetOut.textContent = 'no gain';
      held.textContent = 'this camera has no gain to set - brighten with the exposure or the room\'s light';
      held.classList.remove('full');
      return;
    }
    const gain = light.gain == null ? 'gain ?' : `gain ${light.gain}/${light.gain_max}`;
    const full = light.gain != null && light.gain >= light.gain_max;
    held.textContent = `${gain}${full ? ' (top)' : ''} · bright ${light.bright ?? '?'}`;
    held.classList.toggle('full', full);
  };
}
