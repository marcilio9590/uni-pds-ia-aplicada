importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest");

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENSION = 640;
const CLASS_THRESHOLD = 0.4;
const NMS_IOU_THRESHOLD = 0.45;
const NMS_MAX_OUTPUT = 20;
const TEMPORAL_DISTANCE_PX = 30; // distância para considerar mesma detecção entre frames
const TEMPORAL_REQUIRED_COUNT = 2; // número de frames necessários antes de emitir
let _detectionHistory = []; // [{x,y,count,score}]

let _labels = [];
let _model = null;

async function loadModelAndLabels() {
  await tf.ready();
  _labels = await (await fetch(LABELS_PATH)).json();
  _model = await await tf.loadGraphModel(MODEL_PATH);

  //warmup
  const dummyInput = tf.ones(_model.inputs[0].shape);
  await _model.executeAsync(dummyInput);
  tf.dispose(dummyInput);
  postMessage({ type: "model-loaded" });
}

/**
Pré-processa a imagem para o formato aceito pelo YOLO:
* - tf.browser.fromPixels(): converte ImageBitmap/ImageData para tensor [H, W, 3] 
* - tf.image.resizeBilinear(): redimensiona para [INPUT_DIM, INPUT_DIM]
* - .div(255): normaliza os valores para [0, 1]
* - .expandDims (0): adiciona dimensão batch [1, H, W, 3]
*
* Uso de tf.tidy():
* - Garante que tensores temporários serão descartados automaticamente, 
* evitando vazamento de memória.
**/
function preProcessImage(input) {
  return tf.tidy(() => {
    const image = tf.browser.fromPixels(input);
    return tf.image
      .resizeBilinear(image, [INPUT_MODEL_DIMENSION, INPUT_MODEL_DIMENSION])
      .div(255)
      .expandDims(0);
  });
}

async function runInference(tensor) {
  const out = await _model.executeAsync(tensor);
  tf.dispose(tensor);

  // Assume que as 3 primeiras saidas são: caixas(boxes), pontuações(scores), e classes
  const [boxes, scores, classes] = out.slice(0, 3);
  const [boxesData, scoresData, classesData] = await Promise.all([
    boxes.data(),
    scores.data(),
    classes.data(),
  ]);
  out.forEach((t) => t.dispose());
  return {
    boxes: boxesData,
    scores: scoresData,
    classes: classesData,
  };
}
/**
 * Filtra e processa as predições:
 * - Aplica o limiar de confiança (CLASS_THRESHOLD)
 * - Filtra apenas a classe desejada (exemplo: 'kite') ]
 * - Converte coordenadas normalizadas para pixels reais
 * - Calcula o centro do bounding box
 *
 * Uso de generator (function*):
 * - Permite enviar cada predição assim que processada, sem criar lista intermediária */
async function* processPrediction({ boxes, scores, classes }, width, height) {
  // Collect candidate boxes (apply class & score threshold)
  const candidates = [];
  for (let index = 0; index < scores.length; index++) {
    if (scores[index] < CLASS_THRESHOLD) continue;
    const label = _labels[classes[index]];
    if (label !== "bird" && label !== "kite") continue;
    console.log(label);
    candidates.push({ index, score: scores[index] });
  }

  if (candidates.length === 0) return;

  // Convert boxes to absolute coords and into [y1,x1,y2,x2] for NMS
  const boxesForNMS = [];
  const centers = [];
  for (const c of candidates) {
    let i = c.index;
    let [a, b, c1, d] = boxes.slice(i * 4, (i + 1) * 4);

    // Heuristic: if (c1 < a) or width very small, assume format is [cx, cy, w, h]
    let x1, y1, x2, y2;
    const maybeCx = c1 < a || Math.abs(c1 - a) < 0.01;
    if (maybeCx) {
      const cx = a * width;
      const cy = b * height;
      const bw = c1 * width;
      const bh = d * height;
      x1 = cx - bw / 2;
      x2 = cx + bw / 2;
      y1 = cy - bh / 2;
      y2 = cy + bh / 2;
    } else {
      x1 = a * width;
      y1 = b * height;
      x2 = c1 * width;
      y2 = d * height;
    }

    // clamp
    x1 = Math.max(0, x1);
    y1 = Math.max(0, y1);
    x2 = Math.min(width, x2);
    y2 = Math.min(height, y2);

    boxesForNMS.push([y1, x1, y2, x2]);
    centers.push({
      x: x1 + (x2 - x1) / 2,
      y: y1 + (y2 - y1) / 2,
      score: c.score,
    });
  }

  // Run NMS with TF.js
  const boxesTensor = tf.tensor2d(boxesForNMS);
  const scoresTensor = tf.tensor1d(candidates.map((c) => c.score));
  const selectedIndicesTensor = await tf.image.nonMaxSuppressionAsync(
    boxesTensor,
    scoresTensor,
    NMS_MAX_OUTPUT,
    NMS_IOU_THRESHOLD,
    CLASS_THRESHOLD,
  );
  const selectedIndices = await selectedIndicesTensor.array();
  boxesTensor.dispose();
  scoresTensor.dispose();
  selectedIndicesTensor.dispose();

  // Temporal smoothing: require detection to appear in multiple consecutive frames
  const nextHistory = [];
  for (const si of selectedIndices) {
    const c = centers[si];
    // find matching in previous history
    let matched = null;
    for (let h of _detectionHistory) {
      const dx = h.x - c.x;
      const dy = h.y - c.y;
      if (Math.hypot(dx, dy) <= TEMPORAL_DISTANCE_PX) {
        matched = h;
        break;
      }
    }
    const count = matched && matched.count ? matched.count + 1 : 1;
    nextHistory.push({ x: c.x, y: c.y, count, score: c.score });
    if (count >= TEMPORAL_REQUIRED_COUNT) {
      yield {
        x: c.x,
        y: c.y,
        score: (c.score * 100).toFixed(2),
      };
    }
  }

  // update history (keep only recent)
  _detectionHistory = nextHistory.slice(0, 50);
}

loadModelAndLabels();

self.onmessage = async ({ data }) => {
  if (data.type !== "predict") return;
  if (!_model) return;
  const input = preProcessImage(data.image);
  const { width, height } = data.image;
  const inferenceResults = await runInference(input);

  for await (const prediction of processPrediction(inferenceResults, width, height)) {
    postMessage({
      type: "prediction",
      ...prediction,
    });
  }
};

console.log("🧠 YOLOv5n Web Worker initialized");
