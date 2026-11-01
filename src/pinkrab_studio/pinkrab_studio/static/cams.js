// The things every page that shows the live view's cameras does with them:
// put a picture in a tile, write a number into a card, and say which socket a
// tile's camera comes from.

import { watch } from './live.js';

// The live server is started by the studio and takes a moment to bind; an
// <img> aimed at a port nobody is listening on fails once and stays failed,
// with no event worth retrying on. So a page points a tile at the stream only
// once the live view's numbers have come (the `stats` topic): the studio got
// them from that same server. Each time with a URL of its own: given one it
// has shown before - a camera that dropped and came back - the browser may
// paint the <img> from its image cache, the last frame, without asking the
// server at all, and the picture stays still until the page is reloaded.
export function streamUrl(url) {
  return `${url}${url.includes('?') ? '&' : '?'}at=${Date.now()}`;
}

export function showStream(img, url) {
  img.src = streamUrl(url);
  img.hidden = false;
  img.parentElement.querySelector('.off').hidden = true;
  img.parentElement.classList.add('on');
}

export function hideCam(img) {
  img.src = '';
  img.hidden = true;
  img.parentElement.querySelector('.off').hidden = false;
  img.parentElement.classList.remove('on');
}

export function metric(el, big, sub, state) {
  el.querySelector('.big').textContent = big;
  el.querySelector('.sub').textContent = sub;
  el.className = 'metric' + (state ? ` ${state}` : '');
}


// Which socket a tile's camera comes from, as a select in its caption, kept
// to the cameras on the bus by the `cameras/<which>` topic. The modules report
// the same serial number, so a socket is the only name a camera has; choosing
// one writes the set's file - `which`, the handhelds' cameras.yaml or the
// follower arms' - and starts the live view again if it shows that set, and
// the tile shows the new picture once it is back. `onChanged` is told so the
// page can drop its picture and wait for the view again.
export function sourceSelect(fig, side, onChanged, which = 'handheld') {
  let select = fig.querySelector('figcaption select.source');
  if (!select) {
    select = document.createElement('select');
    select.className = 'source';
    select.title = which === 'follower' ? `which camera is the ${side} arm's` : `which camera is the ${side} gripper's`;
    fig.querySelector('figcaption').append(select);
    // The set's cameras as last sent: put back after a choice the server
    // refused, and after the select is left - neither is a change it sends.
    let cams = null;
    select.addEventListener('change', async () => {
      select.disabled = true;
      let refused = false;
      try {
        const r = await fetch('/cameras', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ side, socket: select.value || null, which }) });
        const answer = await r.json();
        if (answer.error) { alert(answer.error); refused = true; }
      } catch (err) { alert(String(err)); refused = true; }
      select.disabled = false;
      // Refused, the select goes back to what the file says; taken, the new
      // mapping comes down the socket.
      if (refused) fillSources(select, side, cams, true);
      if (onChanged) onChanged();
    });
    select.addEventListener('blur', () => fillSources(select, side, cams));
    watch(`cameras/${which}`, (sent) => { cams = sent; fillSources(select, side, cams); });
  }
  return select;
}

function fillSources(select, side, cams, evenOpen = false) {
  // Left alone while open: options swapped under the pointer take the choice.
  if (!cams?.present || (document.activeElement === select && !evenOpen)) return;
  const mine = cams.present.find((c) => c.side === side);
  const options = [['', 'no camera']].concat(cams.present.map((c) => [c.id, `socket ${(c.port || c.id).replace(/^pci-.*usb-0:/, '').replace(/:1\.0-video-index0$/, '')}` + (c.side && c.side !== side ? ` (${c.side})` : '')]));
  const wanted = options.map(([v, t]) => v + '|' + t).join(';');
  if (select.dataset.options !== wanted) {
    select.replaceChildren(...options.map(([value, text]) => Object.assign(document.createElement('option'), { value, textContent: text })));
    select.dataset.options = wanted;
  }
  select.value = mine ? mine.id : '';
}
