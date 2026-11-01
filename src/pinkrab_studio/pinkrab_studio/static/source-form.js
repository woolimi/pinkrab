// The form a training source - a machine over ssh - is added with, or saved
// again: the Train page's Sources and the Curate page's SAM 3 on. A source
// saved here is one both pages list, the moment it is saved.

const FIELDS = `
  <label>name <input name="name" required placeholder="gpu-box"></label>
  <label>ssh, sftp or scp line <textarea name="line" rows="2" placeholder="ssh -i ~/.ssh/id_ed25519 -p 22 user@gpu.example.com"></textarea></label>
  <label>key file, as a path here <input name="key" placeholder="~/.ssh/id_ed25519"></label>
  <label>or the file itself <input type="file" name="keyfile"></label>
  <p class="hint">Either way the key is copied into <code>~/pinkrab/keys</code>, readable by this user only, and used from there: the file it came from can be moved or deleted. None needed where <code>ssh</code> logs in without one.</p>
  <label>runs are under <input name="root" required value="~/pinkrab-train"></label>
  <p class="hint">Read every 30 s: every run under it that <code>pinkrab-train</code> keeps a <code>metrics.jsonl</code> for. SAM 3 is set up there too, when the Curate page asks for it.</p>
  <p class="error"></p>
  <div class="choice"><button type="submit" class="primary">Save</button></div>`;

async function post(url, body) {
  const answer = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let json = {};
  try { json = await answer.json(); } catch { /* an empty answer */ }
  if (!answer.ok || json.error) throw new Error(json.error || `${answer.status}`);
  return json;
}

// The fields into `form`, saved on submit; `onSaved(name)` is then awaited,
// and what it throws is said in the form. Returns the form's `fill(source)`
// for saving a listed source again, and `say(text)`.
export function sourceForm(form, onSaved = async () => {}) {
  form.classList.add('source-form');
  form.insertAdjacentHTML('beforeend', FIELDS);
  const errorEl = form.querySelector('.error');
  const save = form.querySelector('button[type=submit]');
  const say = (text) => { errorEl.textContent = text; };
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    say('');
    save.disabled = true;
    const { name, line, key, root } = Object.fromEntries(new FormData(form).entries());
    const file = form.keyfile.files[0];
    try {
      await post('/train/sources', { name, line, key, root });
      if (file) await post('/train/sources/key', { name, key: await file.text() });
      say('saved');
      await onSaved(name);
    } catch (exc) {
      say(exc.message);
    } finally {
      save.disabled = false;
    }
  });
  return {
    say,
    fill(s) {
      form.name.value = s.name; form.key.value = s.key || ''; form.root.value = s.root;
      form.line.value = `ssh -p ${s.port} ${s.user || ''}@${s.host}`;
      form.line.focus();
    },
  };
}
