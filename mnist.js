window.addEventListener("DOMContentLoaded", init);

let mouse = { drag: false, x: null, y: null };

function init(){
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 140;
    canvas.height = 140;

    const canvas28 = document.getElementById('canvas28');
    canvas28.width = 28;
    canvas28.height = 28;
    const ctx28 = canvas28.getContext('2d');

    canvas.addEventListener('mousedown', (evt) => {
        drawStart(ctx, evt.offsetX, evt.offsetY);
    });
    canvas.addEventListener('mouseup',  (evt) => {
        drawEnd(ctx, evt.offsetX, evt.offsetY);
    });
    canvas.addEventListener('mouseout', (evt) => {
        drawEnd(ctx, evt.offsetX, evt.offsetY);
    });
    canvas.addEventListener('mousemove', (evt) => {
        draw(ctx, evt.offsetX, evt.offsetY);
    });
    canvas.addEventListener('dblclick', () => {
        clearCanvas(canvas, ctx, canvas28, ctx28);
    });
}


function drawStart(ctx, x, y){
    mouse.drag = true;
    mouse.x = x;
    mouse.y = y;
    // console.log(mouse);

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 10;
    ctx.strokeStyle = 'black';
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function drawEnd(ctx, x, y){
    if (mouse.drag) {
        mouse.drag = false;
        mouse.x = x;
        mouse.y = y;
        // console.log(mouse);

        getCanvas(ctx);
    }
}

function draw(ctx, x, y){
    if (mouse.drag) {
        mouse.x = x;
        mouse.y = y;
        // console.log(mouse);

        ctx.lineTo(x, y);
        ctx.stroke();
        // getCanvas(ctx);
    }
}

function clearCanvas(canvas, ctx, canvas28, ctx28){
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx28.clearRect(0, 0, canvas28.width, canvas28.height);
    reset();
}


function getCanvas(ctx){
    w = 140;
    h = 140;

    //left, top, width, height
    let imageData = ctx.getImageData(0, 0, w, h);

    let img28 = Array(28*28).fill(0)

    const scale = w / 28;  // 5 at w=140

    let col;
    let row;
    let k;
    for (let i = 0; i < imageData.data.length/4; i++) {
        col = Math.trunc( i / w );
        col = Math.trunc( col / scale );
        
        row = Math.trunc( i / scale );
        row = row % 28;
        
        k = col * 28 + row;
        alpha = imageData.data[i*4 + 3];
        img28[k] += alpha;

    }

    img28 = img28.map(x => Math.floor(x / (scale * scale)));  // 0-255
    drawCanvas28(img28)

    img28 = img28.map(x => x/255.0);  // 0-1
    img28 = img28.map(x => (x - 0.1307) / 0.3081);  // 0-1

    runONNX(img28);
}


function reset(){
    console.log("reset")
    let float32 = new Float32Array(10).fill(0);
    console.log(float32);
    display(float32);
}


async function runONNX(array784) {
    try {
        const session = await ort.InferenceSession.create('./mnist.onnx');

        const dataA = Float32Array.from(array784);
        const tensorA = new ort.Tensor('float32', dataA, [1, 1, 28, 28]);

        // prepare feeds. use model input names as keys.
        const feeds = { input: tensorA };

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const dataC = results.output.data;


        let pred = dataC.map(x => Math.exp(x));
        // let total = pred.reduce((sum, element) => sum + element, 0);
        // console.log(pred);
        // console.log(total);
        // console.log("== Finished runONNX");

        display(pred);

        } catch (e) {
          document.write(`failed to inference ONNX model: ${e}.`);
        }
  }


function display(preds){
    for (let i = 0; i < preds.length; i++) {
        let pred = Math.round(preds[i] * 100)
        document.querySelector('#n' + String(i)).style.width = String(pred) + '%';
        document.querySelector('#n' + String(i) + ' span').innerHTML = String(pred) + '%';
    }
}


function drawCanvas28(img28) {
    let imageData28 = new ImageData(28, 28);  //width, height

    for (let i = 0; i < imageData28.data.length/4; i++) {
        imageData28.data[i*4 + 0] = 0;  // Red
        imageData28.data[i*4 + 1] = 0;  // Green
        imageData28.data[i*4 + 2] = 0;  // Blue
        imageData28.data[i*4 + 3] = img28[i];  // Alpha
    }

    const canvas28 = document.getElementById("canvas28");
    const ctx28 = canvas28.getContext('2d');
    ctx28.putImageData(imageData28, 0, 0);
}
