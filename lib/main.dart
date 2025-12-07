import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:file_picker/file_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:permission_handler/permission_handler.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: CamoHome());
  }
}

class CamoHome extends StatefulWidget {
  @override
  _CamoHomeState createState() => _CamoHomeState();
}

class _CamoHomeState extends State<CamoHome> {
  Interpreter? _interp;
  int inputW = 256, inputH = 256; // default — change if your model expects different
  ui.Image? _displayImage;
  List<int>? _maskPixels; // 0 or 255 per pixel
  double _threshold = 0.5;
  String _status = "Idle";

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() => _status = "Loading model...");
    try {
      // Load interpreter from asset
      _interp = await Interpreter.fromAsset('assets/model.tflite');
      // If you need to get input shape from model:
      final inputShapes = _interp!.getInputTensor(0).shape;
      // inputShapes might be [1, H, W, C] or similar:
      if (inputShapes.length >= 3) {
        inputH = inputShapes[inputShapes.length - 3];
        inputW = inputShapes[inputShapes.length - 2];
      }
      setState(() => _status = "Model loaded (in:${inputW}x${inputH})");
    } catch (e) {
      setState(() => _status = "Model load failed: $e");
    }
  }

  Future<void> _pickAndRun() async {
    // ensure storage permission
    if (!(await Permission.storage.request().isGranted)) {
      setState(() => _status = "Storage permission required");
      return;
    }

    final result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result == null) return;
    final path = result.files.single.path!;
    final file = File(path);
    await _runOnImage(file);
  }

  Future<void> _runOnImage(File file) async {
    setState(() {
      _status = "Decoding image...";
      _maskPixels = null;
      _displayImage = null;
    });

    final bytes = await file.readAsBytes();
    final src = img.decodeImage(bytes);
    if (src == null) {
      setState(() => _status = "Can't decode image");
      return;
    }

    // Save a displayable ui.Image (original or scaled to screen)
    final uiImg = await _toUiImage(src);
    setState(() => _displayImage = uiImg);

    // Resize to model input size
    final resized = img.copyResize(src, width: inputW, height: inputH);

    // Convert to float32 input (normalized 0..1) — adjust if model expects different
    final input = _imageToFloat32List(resized);

    setState(() => _status = "Running inference...");

    // Prepare output buffer — many segmentation models output [1, H, W, 1] or [1, H*W]
    // We'll infer general shape from interpreter output tensor
    // Try a float output sized [1, H, W, 1]
    final outputShape = _interp!.getOutputTensor(0).shape; // e.g. [1,256,256,1]
    final outType = _interp!.getOutputTensor(0).type;

    // prepare output as Float32List of required size
    int outCount = 1;
    for (var d in outputShape) outCount *= d;
    var output = List.filled(outCount, 0.0).reshape([outCount]); // helper below

    // Run
    try {
      // Interpreter.run requires proper shaped input; create typed buffers
      final inputTensor = [input]; // wrapper if model expects batch dim
      // Some models prefer direct Float32List, some prefer nested lists. We attempt direct:
      _interp!.run(input, output);
    } catch (e) {
      // fallback: try runForMultipleInputs
      try {
        final outMap = <int, Object>{0: output};
        _interp!.runForMultipleInputs([input], outMap);
      } catch (e2) {
        setState(() => _status = "Inference failed: $e2");
        return;
      }
    }

    // Convert output to mask pixels (assuming values in 0..1)
    List<int> mask = List.filled(inputW * inputH, 0);
    for (int i = 0; i < inputW * inputH && i < output.length; i++) {
      double val = (output[i]).toDouble();
      mask[i] = (val >= _threshold) ? 255 : 0; // 255 = visible pixel
    }

    setState(() {
      _maskPixels = mask;
      _status = "Done";
    });
  }

  // Helper: convert image.Image to ui.Image to display in Widget
  Future<ui.Image> _toUiImage(img.Image image) async {
    final png = img.encodePng(image);
    final codec = await ui.instantiateImageCodec(Uint8List.fromList(png));
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  // Convert image.Image to Float32List normalized 0..1 (shape: [1,H,W,3])
  Float32List _imageToFloat32List(img.Image src) {
    final int width = src.width;
    final int height = src.height;
    final Float32List floats = Float32List(width * height * 3);
    int idx = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final c = src.getPixel(x, y);
        // image package stores pixels ARGB
        final r = img.getRed(c) / 255.0;
        final g = img.getGreen(c) / 255.0;
        final b = img.getBlue(c) / 255.0;
        floats[idx++] = r;
        floats[idx++] = g;
        floats[idx++] = b;
      }
    }
    return floats;
  }

  @override
  void dispose() {
    _interp?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final display = _displayImage;
    return Scaffold(
      appBar: AppBar(title: Text('Camo - Inference Demo')),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: display == null
                  ? Text(_status)
                  : Stack(
                      children: [
                        // Show original image scaled to fit
                        Positioned.fill(
                          child: FittedBox(
                            fit: BoxFit.contain,
                            child: SizedBox(
                              width: display.width.toDouble(),
                              height: display.height.toDouble(),
                              child: RawImage(image: display),
                            ),
                          ),
                        ),
                        // Overlay mask painter
                        if (_maskPixels != null)
                          Positioned.fill(
                            child: IgnorePointer(
                              child: CustomPaint(
                                painter: MaskPainter(
                                  maskPixels: _maskPixels!,
                                  maskWidth: inputW,
                                  maskHeight: inputH,
                                  color: Colors.red.withOpacity(0.35),
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
            ),
          ),

          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                ElevatedButton(
                  onPressed: _pickAndRun,
                  child: Text("Pick image & run"),
                ),
                SizedBox(width: 12),
                Text("Threshold: ${_threshold.toStringAsFixed(2)}"),
                Expanded(
                  child: Slider(
                    value: _threshold,
                    min: 0.0,
                    max: 1.0,
                    onChanged: (v) => setState(() => _threshold = v),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Simple painter that scales raw mask (inputW x inputH) to widget size and paints semi-transparent rects
class MaskPainter extends CustomPainter {
  final List<int> maskPixels; // 0 or 255 values, length = maskW * maskH
  final int maskWidth;
  final int maskHeight;
  final Color color;

  MaskPainter({
    required this.maskPixels,
    required this.maskWidth,
    required this.maskHeight,
    required this.color,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..style = PaintingStyle.fill;
    final double cellW = size.width / maskWidth;
    final double cellH = size.height / maskHeight;

    for (int y = 0; y < maskHeight; y++) {
      for (int x = 0; x < maskWidth; x++) {
        int idx = y * maskWidth + x;
        if (idx >= maskPixels.length) continue;
        if (maskPixels[idx] != 0) {
          paint.color = color;
          final rect = Rect.fromLTWH(x * cellW, y * cellH, cellW, cellH);
          canvas.drawRect(rect, paint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant MaskPainter other) => other.maskPixels != maskPixels;
}

// extension to reshape list (quick helper)
extension ListReshape on List {
  List reshape(List<int> dims) {
    // no-op in this simplified example
    return this;
  }
}

