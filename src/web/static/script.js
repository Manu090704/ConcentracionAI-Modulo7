const input = document.getElementById("file");
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const errorBox = document.getElementById("error");
const result = document.getElementById("result");
const verdict = document.getElementById("verdict");
const original = document.getElementById("original");
const processed = document.getElementById("processed");
const bars = document.getElementById("bars");

const IDLE_TEXT = "Haz clic o arrastra una imagen aquí";
let originalUrl = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  result.hidden = true;
}

function renderBars(probabilities) {
  bars.replaceChildren();
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  for (const [name, value] of entries) {
    const item = document.createElement("li");

    const label = document.createElement("span");
    label.textContent = name;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(value * 100).toFixed(1)}%`;
    track.append(fill);

    const percent = document.createElement("span");
    percent.className = "value";
    percent.textContent = `${(value * 100).toFixed(1)}%`;

    item.append(label, track, percent);
    bars.append(item);
  }
}

function render(data, fileUrl) {
  const confidence = data.probabilidades[data.clase];
  verdict.textContent = `Predicción: ${data.clase} (${(confidence * 100).toFixed(1)}%)`;
  original.src = fileUrl;
  processed.src = data.imagen_procesada;
  renderBars(data.probabilidades);
  result.hidden = false;
}

async function classify(file) {
  errorBox.hidden = true;
  if (!file.type.startsWith("image/")) {
    showError("El archivo seleccionado no es una imagen.");
    return;
  }

  dropzone.classList.add("busy");
  dropzoneText.textContent = "Analizando…";

  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);

  const form = new FormData();
  form.append("imagen", file);

  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "No se pudo procesar la imagen.");
    render(data, originalUrl);
  } catch (err) {
    showError(err instanceof SyntaxError || err instanceof TypeError
      ? "No se pudo conectar con el servidor."
      : err.message);
  } finally {
    dropzone.classList.remove("busy");
    dropzoneText.textContent = IDLE_TEXT;
  }
}

input.addEventListener("change", () => {
  if (input.files.length > 0) classify(input.files[0]);
  input.value = "";   // permite volver a elegir el mismo archivo
});

for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
}
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) classify(file);
});
