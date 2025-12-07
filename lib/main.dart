import 'package:flutter/material.dart';

void main() => runApp(const GaceApp());

class GaceApp extends StatelessWidget {
  const GaceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GACE Camo Eval',
      home: Scaffold(
        appBar: AppBar(
          title: const Text('GACE Camo Eval (demo)'),
        ),
        body: const Center(
          child: Padding(
            padding: EdgeInsets.all(20),
            child: Text(
              'APK smoke test build.\\n\\n'
              'Model file: assets/model.tflite\\n'
              'Once this installs and runs on your phone,\\n'
              'we will wire real inference + overlay.',
              textAlign: TextAlign.center,
            ),
          ),
        ),
      ),
    );
  }
}
