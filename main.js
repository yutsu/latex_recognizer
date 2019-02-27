/* variables */
var model;
var canvas;
var latexSymbols = [];
var canvas;
var coords = [];
var mousePressed = false;





/* setup drawing canvas */
$(function() {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 0;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 8;
    canvas.renderAll();
    //setup listeners
    canvas.on('mouse:up', () => {
        getPrediction();
        mousePressed = false
    });
    canvas.on('mouse:down', () => mousePressed = true);
    canvas.on('mouse:move', (e) => saveCoor(e));
})

/* Display the prediction result */
function displayResult(results, probs) {
    //loop over the predictions
    for (var i = 0; i < results.length; i++) {
        $('#result-title').html('Result')
        $('#symbol' + (i+1)).html('$'+results[i]+'$')
        $('#latex' + (i+1)).html(results[i])
        // $('#prob' + (i+1)).html(Math.round(probs[i] * 100) + '%')
        barGraph(probs[i], i)
    }
    //render MathJax
   MathJax.Hub.Queue(["Typeset",MathJax.Hub,this.formula]);
}

/* horizontal bar graph according to probabilities */
function barGraph(prob, i) {
    const colors = ['#ED6A5A', '#F4F1BC', '#A2C4BF', '#7CBABF', '#E6EBE0']
    let symbol = document.getElementById('symbol' + (i + 1))
    symbol.style.backgroundImage = 'linear-gradient(' + colors[i] +', ' +colors[i] + ')';
    symbol.style.backgroundSize = Math.round(prob * 100) + '%' + ' 15%'
}


// save drawing coordinates
function saveCoor(event) {
    const pointer = canvas.getPointer(event.e);
    const posX = pointer.x;
    const posY = pointer.y;

    if (posX >= 0 && posY >= 0 && mousePressed) {
        coords.push(pointer)
    }
}


/* get the current image data */
function getImageData() {
        // find the bounding box of free drawing
        const coorX = coords.map(p => p.x);
        const coorY = coords.map(p => p.y);

        //find top left and bottom right corners
        const min_coords = {
            x: Math.min.apply(null, coorX),
            y: Math.min.apply(null, coorY)
        }
        const max_coords = {
            x: Math.max.apply(null, coorX),
            y: Math.max.apply(null, coorY)
        }

        const dpi = window.devicePixelRatio

        //crop the image
        const imgData = canvas.contextContainer.getImageData(min_coords.x * dpi, min_coords.y * dpi, (max_coords.x - min_coords.x) * dpi, (max_coords.y - min_coords.y) * dpi);

        /* without crop */
        // const imgData = canvas.contextContainer.getImageData(0, 0, 300* dpi, 300* dpi)
        return imgData
    }


// get the prediction through model
function getPrediction() {
    //make sure we have at least two recorded coordinates
    if (coords.length >= 2 && model ) {

        //get the image data from the canvas
        const imgData = getImageData()

        //get the prediction
        const pred = model.predict(preprocess(imgData)).dataSync()

        //find the top 5 predictions
        const indices = findIndicesOfBestPred(pred, 5)
        const probs = findBestProb(pred, 5)
        const results = getLabels(indices)

        //set the table
        displayResult(results, probs)
    }

}

// get the labels (japanese characters)
function getLabels(indices) {
    let output = []
    for (let i = 0; i < indices.length; i++)
        output[i] = latexSymbols[indices[i]]
    return output
}

/* load Japanese character labels */
async function loadLabel() {
    loc = 'latex_labels.txt'

    await $.ajax({
        url: loc,
        dataType: 'text',
    }).done(success);
}
function success(data) {
    const labels = data.split(/\n/)
    for (var i = 0; i < labels.length - 1; i++) {
        let ch = labels[i]
        latexSymbols[i] = ch
    }
}

/* get indices of the best (count) probs */
function findIndicesOfBestPred(pred, count) {
    let indices = [];
    for (let i = 0; i < pred.length; i++) {
        indices.push(i);
        if (indices.length > count) {
            indices.sort((a, b) => pred[b] - pred[a]);
            indices.pop();
        }
    }
    return indices;
}

/* find the best (count) prediction probabilities */
function findBestProb(pred, count) {
    let probs = [];
    let indices = findIndicesOfBestPred(pred, count)
    for (let i = 0; i < indices.length; i++)
        probs[i] = pred[indices[i]]
    return probs
}

/* preprocess the data */
function preprocess(imgData) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgData, numChannels = 3)
        const resized = tf.image.resizeBilinear(tensor, [32, 32]).toFloat()
        const rgb = tf.scalar(255.0);
        const normalized = resized.div(rgb);
        // add an additional dimension at index 0 to get a batch shape
        const processed = normalized.expandDims(0)
        return processed
    })
}

/* load the model */
async function startLoadingModel() {
    model = await tf.loadModel('model/model.json')
    //warm up
    model.predict(tf.zeros([1, 32, 32, 3]))

    /* allow drawing on canvas */
    allowDrawing()
    await loadLabel()
}

/* allow drawing on canvas */
function allowDrawing() {
    canvas.isDrawingMode = 1;
    document.getElementById('status').innerHTML = 'Ready!';

    $('button').prop('disabled', false);
}

/* clear the canvas onclick */
function eraseCanvas() {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    coords = [];
}
