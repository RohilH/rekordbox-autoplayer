const $ = (id) => document.getElementById(id);

let libraryRoot = null;
let shownPlaylist = null;
let lastState = {};
let settingsFocused = false;

function formatTime(seconds) {
  if (seconds == null || !isFinite(seconds)) return "0:00";
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error((await response.json()).detail ?? response.statusText);
  return response.json();
}

function renderTree(nodes) {
  const ul = document.createElement("ul");
  for (const node of nodes) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    if (node.is_folder) {
      li.className = "folder";
      button.textContent = `▸ ${node.name}`;
      const children = renderTree(node.children);
      children.hidden = true;
      button.addEventListener("click", () => {
        children.hidden = !children.hidden;
        button.textContent = `${children.hidden ? "▸" : "▾"} ${node.name}`;
      });
      li.append(button, children);
    } else {
      button.dataset.playlistId = node.id;
      button.textContent = node.name;
      const count = document.createElement("span");
      count.className = "count";
      count.textContent = node.track_count;
      button.append(count);
      button.addEventListener("click", () => showPlaylist(node.id));
      li.append(button);
    }
    ul.append(li);
  }
  return ul;
}

async function loadLibrary() {
  const library = await api("GET", "/api/library");
  const usb = $("usb");
  if (!library.connected) {
    usb.textContent = library.error ?? "No rekordbox USB found";
    usb.className = "pill";
    if (libraryRoot !== null) {
      libraryRoot = null;
      $("playlists").replaceChildren();
    }
    return;
  }
  usb.textContent = library.name;
  usb.className = "pill ok";
  if (library.root === libraryRoot) return;
  libraryRoot = library.root;
  $("playlists").replaceChildren(renderTree(library.playlists));
}

async function showPlaylist(id) {
  const playlist = await api("GET", `/api/playlists/${id}`);
  shownPlaylist = playlist;
  document.querySelectorAll("nav button.selected").forEach((b) => b.classList.remove("selected"));
  document.querySelector(`nav button[data-playlist-id="${id}"]`)?.classList.add("selected");
  $("playlist-name").textContent = playlist.name;
  $("play-all").hidden = playlist.tracks.length === 0;
  $("empty").hidden = playlist.tracks.length > 0;
  $("empty").textContent = "This playlist is empty.";
  $("tracks").hidden = playlist.tracks.length === 0;

  const rows = playlist.tracks.map((track, index) => {
    const tr = document.createElement("tr");
    tr.dataset.index = index;
    const cells = [index + 1, track.title, track.artist, track.bpm.toFixed(1), track.key, formatTime(track.duration_s)];
    cells.forEach((value, i) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (i === 3 || i === 5) td.className = "num";
      tr.append(td);
    });
    tr.addEventListener("click", () => play(index));
    return tr;
  });
  $("tracks").querySelector("tbody").replaceChildren(...rows);
  highlightRows(lastState);
}

function play(index) {
  if (shownPlaylist) api("POST", "/api/play", { playlist_id: shownPlaylist.id, index });
}

function highlightRows(state) {
  const samePlaylist = shownPlaylist && state.playlist_id === shownPlaylist.id;
  const failed = new Set(samePlaylist ? state.failed ?? [] : []);
  document.querySelectorAll("#tracks tbody tr").forEach((tr) => {
    const index = Number(tr.dataset.index);
    tr.classList.toggle("current", samePlaylist && index === state.index);
    tr.classList.toggle("next", samePlaylist && index === state.next_index);
    tr.classList.toggle("failed", failed.has(index));
  });
}

function mixStatus(state) {
  const onCues = (cues) => (cues ? ` on cues ${cues}` : "");
  if (state.mixing) {
    const text = state.beatmatched ? `Beat-matched mix${onCues(state.cues)}` : "Crossfading";
    return { text, mixing: true };
  }
  if (state.mix_in_s != null) {
    const kind = {
      beatmatch: state.next_beatmatched ? `Beat-matched mix${onCues(state.next_cues)}` : "Crossfade",
      timed: "Crossfade",
      off: "Next track",
    };
    return { text: `${kind[state.settings.mode]} in ${formatTime(state.mix_in_s)}`, mixing: false };
  }
  if (state.loading) return { text: "Loading next track…", mixing: false };
  return { text: "", mixing: false };
}

function renderState(state) {
  lastState = state;
  const active = state.status === "playing" || state.status === "paused";
  $("toggle").disabled = !active;
  $("skip").disabled = !active;
  $("stop").disabled = state.status === "idle";
  const toggleLabel = state.paused ? "Resume" : "Pause";
  $("toggle").classList.toggle("paused", Boolean(state.paused));
  $("toggle").setAttribute("aria-label", toggleLabel);
  $("toggle").title = toggleLabel;
  $("output-device").textContent = state.output_device ? `Output: ${state.output_device}` : "";
  renderTempo(state);

  if (state.track) {
    $("now-title").textContent = `${state.track.title} — ${state.track.artist}`;
    const change = state.tempo_change;
    const tempo = Math.abs(change) >= 0.0005
      ? ` · tempo ${change > 0 ? "+" : ""}${(change * 100).toFixed(1)}%${state.key_locked ? " (master tempo)" : ""}`
      : "";
    $("now-meta").textContent = `${state.playlist_name} · ${state.index + 1} · ${state.track.bpm.toFixed(1)} BPM${tempo}`;
    $("pos").textContent = formatTime(state.position_s);
    $("dur").textContent = formatTime(state.duration_s);
    $("progress-bar").style.width = `${Math.min(100, (100 * state.position_s) / state.duration_s)}%`;
    renderZones(state);
  } else {
    $("now-title").textContent = { loading: "Loading…", finished: "Playlist finished" }[state.status] ?? "Nothing playing";
    $("now-meta").textContent = "";
    $("pos").textContent = $("dur").textContent = "0:00";
    $("progress-bar").style.width = "0";
    renderZones({});
  }
  $("seek").classList.toggle("disabled", !state.track);
  const mix = mixStatus(state);
  $("mix-status").textContent = mix.text;
  $("mix-status").className = mix.mixing ? "mixing" : "";

  if (!settingsFocused) {
    const form = $("settings");
    form.mode.value = state.settings.mode;
    form.overlap_bars.value = state.settings.overlap_bars;
    form.ease_bars.value = state.settings.ease_bars;
    form.key_lock.checked = state.settings.key_lock;
  }
  highlightRows(state);
}

const BPM_MIN = 40;
const BPM_MAX = 300;
const BPM_SEND_INTERVAL_MS = 120;
const BPM_REPEAT_DELAY_MS = 350;
const BPM_REPEAT_EVERY_MS = 70;

// The BPM the listener asked for, shown at once and sent at most every BPM_SEND_INTERVAL_MS, so rapid
// clicks or a held button are never limited by the round trip to the player.
let bpmTarget = null;
let bpmLastSent = 0;
let bpmSendTimer = null;

const clampBpm = (bpm) => Math.min(BPM_MAX, Math.max(BPM_MIN, Math.round(bpm * 10) / 10));

function sendBpm() {
  bpmSendTimer = null;
  bpmLastSent = performance.now();
  api("PUT", "/api/master-bpm", { bpm: bpmTarget }).catch(() => {});
}

function setBpm(bpm) {
  bpmTarget = bpm == null ? null : clampBpm(bpm);
  renderTempo(lastState);
  if (bpmSendTimer) return;
  const wait = BPM_SEND_INTERVAL_MS - (performance.now() - bpmLastSent);
  if (wait <= 0) sendBpm();
  else bpmSendTimer = setTimeout(sendBpm, wait);
}

function currentBpm() {
  if (bpmTarget != null) return bpmTarget;
  if (lastState.master_bpm != null) return lastState.master_bpm;
  return lastState.bpm != null ? Math.round(lastState.bpm) : null;
}

function nudgeBpm(delta) {
  const base = currentBpm();
  if (base != null) setBpm(base + delta);
}

function renderTempo(state) {
  const settled = bpmSendTimer == null && performance.now() - bpmLastSent > 600;
  if (bpmTarget != null && settled && state.master_bpm === bpmTarget) bpmTarget = null;
  const master = bpmTarget ?? state.master_bpm;
  const hasTrack = Boolean(state.track) && state.bpm != null;
  $("bpm-down").disabled = $("bpm-up").disabled = !hasTrack;
  $("bpm-input").disabled = !hasTrack;
  const input = $("bpm-input");
  if (document.activeElement !== input) {
    const shown = bpmTarget ?? (hasTrack ? state.bpm : master);
    input.value = shown != null ? shown.toFixed(1) : "—";
  }
  $("bpm-label").textContent = master != null ? `Master ${Number(master.toFixed(1))}` : "BPM";
  $("bpm-reset").hidden = master == null;
  document.querySelector(".tempo").classList.toggle("master", master != null);
}

function holdToRepeat(button, delta) {
  let delay = null;
  let repeat = null;
  const stop = () => {
    clearTimeout(delay);
    clearInterval(repeat);
    delay = repeat = null;
  };
  button.addEventListener("pointerdown", (event) => {
    if (button.disabled || event.button !== 0) return;
    nudgeBpm(delta);
    try {
      button.setPointerCapture(event.pointerId);
    } catch {
      // Without capture, leaving the button still ends the repeat via pointerleave.
    }
    delay = setTimeout(() => (repeat = setInterval(() => nudgeBpm(delta), BPM_REPEAT_EVERY_MS)), BPM_REPEAT_DELAY_MS);
  });
  for (const type of ["pointerup", "pointercancel", "pointerleave", "lostpointercapture"]) button.addEventListener(type, stop);
  button.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      nudgeBpm(delta);
    }
  });
}

holdToRepeat($("bpm-up"), 1);
holdToRepeat($("bpm-down"), -1);
$("bpm-reset").addEventListener("click", () => setBpm(null));

const bpmInput = $("bpm-input");
bpmInput.addEventListener("focus", () => bpmInput.select());
bpmInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    const typed = parseFloat(bpmInput.value);
    if (Number.isFinite(typed)) setBpm(typed);
    bpmInput.blur();
  } else if (event.key === "Escape") {
    bpmInput.blur();
  } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
    event.preventDefault();
    nudgeBpm(event.key === "ArrowUp" ? 1 : -1);
    bpmInput.value = currentBpm().toFixed(1);
  }
});
bpmInput.addEventListener("blur", () => renderTempo(lastState));

document.addEventListener("keydown", (event) => {
  const typing = event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement;
  if (typing || event.metaKey || event.ctrlKey || !lastState.track) return;
  if (event.key === "+" || event.key === "=") nudgeBpm(1);
  else if (event.key === "-" || event.key === "_") nudgeBpm(-1);
});

function setZone(id, from, to, duration) {
  const zone = $(id);
  if (from == null || to == null || !duration || to <= from) {
    zone.style.width = "0";
    return;
  }
  zone.style.left = `${(100 * from) / duration}%`;
  zone.style.width = `${(100 * (to - from)) / duration}%`;
}

function renderZones(state) {
  const d = state.duration_s;
  setZone("zone-skipped-start", 0, state.play_start_s, d);
  setZone("zone-mix-in", state.play_start_s, state.mix_in_end_s, d);
  setZone("zone-mix-out", state.mix_out_start_s, state.play_end_s, d);
  setZone("zone-skipped-end", state.play_end_s, d, d);
}

function seekFraction(event) {
  const rect = $("seek").querySelector(".progress").getBoundingClientRect();
  return Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
}

function seekTo(seconds) {
  if (!lastState.track) return;
  const position = Math.min(Math.max(seconds, 0), lastState.duration_s);
  $("progress-bar").style.width = `${(100 * position) / lastState.duration_s}%`;
  api("POST", "/api/seek", { position_s: position });
}

const seekBar = $("seek");
seekBar.addEventListener("click", (event) => seekTo(seekFraction(event) * lastState.duration_s));
seekBar.addEventListener("mousemove", (event) => {
  seekBar.title = lastState.track ? formatTime(seekFraction(event) * lastState.duration_s) : "";
});
seekBar.addEventListener("keydown", (event) => {
  const step = { ArrowLeft: -10, ArrowRight: 10 }[event.key];
  if (step == null || !lastState.track) return;
  event.preventDefault();
  seekTo(lastState.position_s + step);
});

function connect() {
  const socket = new WebSocket(`ws://${location.host}/api/state`);
  socket.onmessage = (event) => renderState(JSON.parse(event.data));
  socket.onclose = () => setTimeout(connect, 1000);
}

$("play-all").addEventListener("click", () => play(0));
$("toggle").addEventListener("click", () => api("POST", lastState.paused ? "/api/resume" : "/api/pause"));
$("skip").addEventListener("click", () => api("POST", "/api/skip"));
$("stop").addEventListener("click", () => api("POST", "/api/stop"));

const settingsForm = $("settings");
settingsForm.addEventListener("focusin", () => (settingsFocused = true));
settingsForm.addEventListener("focusout", () => (settingsFocused = false));
settingsForm.addEventListener("change", () => {
  api("PUT", "/api/settings", {
    mode: settingsForm.mode.value,
    overlap_bars: Number(settingsForm.overlap_bars.value),
    ease_bars: Number(settingsForm.ease_bars.value),
    key_lock: settingsForm.key_lock.checked,
  });
});

loadLibrary();
setInterval(loadLibrary, 3000);
connect();
