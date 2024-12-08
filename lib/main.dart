import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'dart:typed_data';
import 'lane_detector.dart';
import 'guidance_service.dart';
import 'lane_guide_painter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(camera: cameras.first));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  
  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Blind Runner Guide',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: LaneDetectionScreen(camera: camera),
    );
  }
}

class LaneDetectionScreen extends StatefulWidget {
  final CameraDescription camera;

  const LaneDetectionScreen({super.key, required this.camera});

  @override
  State<LaneDetectionScreen> createState() => _LaneDetectionScreenState();
}

class _LaneDetectionScreenState extends State<LaneDetectionScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool isProcessing = false;
  final GuidanceService _guidanceService = GuidanceService();
  double _currentDeviation = 0.0;
  Timer? _processingTimer;
  Uint8List? _processedImage;
  bool _isLaneDetected = false;
  bool _isRunning = false;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    _initializeControllerFuture = _controller.initialize().then((_) {
      if (mounted) {
        _startImageProcessing();
      }
    });
  }

  void _startImageProcessing() {
    if (_isRunning) return;
    _isRunning = true;
    LaneDetector.clearCache();
    
    _processingTimer = Timer.periodic(
      const Duration(milliseconds: 150),
      (_) async {
        if (!isProcessing && mounted && _isRunning) {
          await _processCurrentFrame();
        }
      }
    );
  }

  void _stopImageProcessing() {
    _isRunning = false;
    _processingTimer?.cancel();
    _processingTimer = null;
    _guidanceService.stop();
    
    if (mounted) {
      setState(() {
        _currentDeviation = 0.0;
        _processedImage = null;
        _isLaneDetected = false;
      });
    }
  }

  Future<void> _processCurrentFrame() async {
    if (!_controller.value.isInitialized) return;
    
    isProcessing = true;
    try {
      final image = await _controller.takePicture();
      final imageBytes = await image.readAsBytes();
      
      final result = await LaneDetector.processFrame(imageBytes);
      
      if (mounted) {
        setState(() {
          _currentDeviation = result.deviation;
          _processedImage = result.processedImage;
          _isLaneDetected = result.isLaneDetected;
        });
        if (_isLaneDetected) {
          await _guidanceService.provideGuidance(result.deviation);
        }
      }
    } catch (e) {
      debugPrint('Error processing frame: $e');
    } finally {
      isProcessing = false;
    }
  }

  @override
  void dispose() {
    _stopImageProcessing();
    LaneDetector.dispose();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Lane Detection Guide'),
        actions: [
          IconButton(
            icon: Icon(_isRunning ? Icons.stop : Icons.play_arrow),
            onPressed: () {
              setState(() {
                if (_isRunning) {
                  _stopImageProcessing();
                } else {
                  _startImageProcessing();
                }
              });
            },
          ),
        ],
      ),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Column(
              children: [
                Expanded(
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      _processedImage != null
                          ? Image.memory(
                              _processedImage!,
                              fit: BoxFit.cover,
                            )
                          : CameraPreview(_controller),
                      CustomPaint(
                        painter: LaneGuidePainter(_currentDeviation),
                      ),
                    ],
                  ),
                ),
                Container(
                  height: 100,
                  color: Colors.black87,
                  child: Center(
                    child: Text(
                      _getGuidanceText(),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                      ),
                    ),
                  ),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }

  String _getGuidanceText() {
    if (_currentDeviation.abs() < 0.1) {
      return 'Centered';
    } else if (_currentDeviation > 0) {
      return 'Move Left';
    } else {
      return 'Move Right';
    }
  }
}
