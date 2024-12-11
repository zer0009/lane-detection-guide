import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'lane_detector.dart';
import 'guidance_service.dart';
import 'lane_guide_painter.dart';

class Constants {
  static const processingInterval = Duration(milliseconds: 100); // Faster processing
  static const deviationThreshold = 0.1;
  static const guidanceTextSize = 24.0;
  static const maxFPS = 30; // Limit FPS for better performance
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Lock to portrait orientation
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  
  final cameras = await availableCameras();
  runApp(MyApp(camera: cameras.first));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  
  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Lane Detection Guide',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.dark(
          primary: Colors.blue,
          secondary: Colors.blueAccent,
        ),
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

class _LaneDetectionScreenState extends State<LaneDetectionScreen> with WidgetsBindingObserver {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool _isProcessing = false;
  final GuidanceService _guidanceService = GuidanceService();
  double _currentDeviation = 0.0;
  Timer? _processingTimer;
  Uint8List? _processedImage;
  bool _isLaneDetected = false;
  bool _isRunning = false;
  DateTime? _lastFrameTime;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    _initializeControllerFuture = _controller.initialize().then((_) {
      if (mounted) {
        _configureCameraSettings();
        _startImageProcessing();
      }
    }).catchError((error) {
      print('Camera initialization error: $error');
    });
  }

  void _startImageProcessing() {
    if (_isRunning) return;
    _isRunning = true;
    LaneDetector.clearCache();
    
    _processingTimer = Timer.periodic(
      Constants.processingInterval,
      (_) => _processNextFrameIfReady()
    );
  }

  Future<void> _processNextFrameIfReady() async {
    // Implement frame rate limiting
    if (_lastFrameTime != null) {
      final elapsed = DateTime.now().difference(_lastFrameTime!);
      if (elapsed.inMilliseconds < (1000 / Constants.maxFPS)) {
        return;
      }
    }
    
    if (!_isProcessing && mounted && _isRunning) {
      await _processCurrentFrame();
    }
  }

  Future<void> _processCurrentFrame() async {
    if (!_controller.value.isInitialized) return;
    
    _isProcessing = true;
    _lastFrameTime = DateTime.now();
    
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
          _provideHapticFeedback(result.deviation);
        }
      }
    } catch (e) {
      debugPrint('Error processing frame: $e');
    } finally {
      _isProcessing = false;
    }
  }

  void _provideHapticFeedback(double deviation) {
    if (deviation.abs() > Constants.deviationThreshold) {
      HapticFeedback.mediumImpact();
    }
  }

  void _configureCameraSettings() {
    _controller.setFlashMode(FlashMode.off);
    _controller.setExposureMode(ExposureMode.auto);
    _controller.setFocusMode(FocusMode.auto);
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

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller.value.isInitialized) {
      switch (state) {
        case AppLifecycleState.paused:
          _stopImageProcessing();
          break;
        case AppLifecycleState.resumed:
          _startImageProcessing();
          break;
        default:
          break;
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopImageProcessing();
    LaneDetector.dispose();
    _controller.dispose();
    _guidanceService.dispose();
    _processedImage = null;
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
            return _buildMainContent();
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }

  Widget _buildMainContent() {
    return Column(
      children: [
        Expanded(
          child: Stack(
            fit: StackFit.expand,
            children: [
              _buildImageDisplay(),
              CustomPaint(
                painter: LaneGuidePainter(_currentDeviation),
              ),
              if (_isProcessing)
                const Positioned(
                  top: 10,
                  right: 10,
                  child: CircularProgressIndicator(),
                ),
            ],
          ),
        ),
        _buildGuidanceBar(),
      ],
    );
  }

  Widget _buildImageDisplay() {
    if (_processedImage != null) {
      return Image.memory(
        _processedImage!,
        fit: BoxFit.cover,
        gaplessPlayback: true, // Prevents flickering
      );
    }
    return CameraPreview(_controller);
  }

  Widget _buildGuidanceBar() {
    return Container(
      height: 100,
      color: Colors.black87,
      child: Center(
        child: Text(
          _getGuidanceText(),
          style: const TextStyle(
            color: Colors.white,
            fontSize: Constants.guidanceTextSize,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }

  String _getGuidanceText() {
    if (!_isLaneDetected) {
      return 'No Lane Detected';
    }
    if (_currentDeviation.abs() < Constants.deviationThreshold) {
      return 'Centered';
    } else {
      return _currentDeviation > 0 ? 'Move Left' : 'Move Right';
    }
  }
}