import 'package:flutter_tts/flutter_tts.dart';
import 'dart:async';

class GuidanceService {
  final FlutterTts tts = FlutterTts();
  DateTime? lastGuidance;
  String? lastMessage;
  Timer? _cooldownTimer;
  bool _isReady = true;
  
  // Add thresholds for different guidance levels
  static const double _CENTER_THRESHOLD = 0.05;
  static const double _SLIGHT_THRESHOLD = 0.15;
  static const double _MODERATE_THRESHOLD = 0.3;
  
  // Track consecutive similar readings for stability
  int _consecutiveSimilarReadings = 0;
  static const int _REQUIRED_SIMILAR_READINGS = 2;
  double? _lastDeviation;

  GuidanceService() {
    _initTTS();
  }

  Future<void> _initTTS() async {
    await tts.setLanguage('en-US');
    await tts.setSpeechRate(1.0);  // Slightly faster for more responsive guidance
    await tts.setVolume(1.0);
    await tts.setPitch(1.0);
    
    // Add error handling
    tts.setErrorHandler((msg) {
      print("TTS Error: $msg");
      _isReady = true;
    });
    
    // Reset ready state after completion
    tts.setCompletionHandler(() {
      _isReady = true;
    });
  }

  Future<void> provideGuidance(double deviation) async {
    if (!_isReady) return;

    // Check for stability in readings
    if (_lastDeviation != null && 
        (deviation - _lastDeviation!).abs() < 0.05) {
      _consecutiveSimilarReadings++;
    } else {
      _consecutiveSimilarReadings = 0;
    }
    _lastDeviation = deviation;

    // Only provide guidance if readings are stable
    if (_consecutiveSimilarReadings < _REQUIRED_SIMILAR_READINGS) return;

    String message = _determineGuidanceMessage(deviation);
    
    // Only speak if message changed or enough time has passed
    if (message != lastMessage || 
        (lastGuidance != null && 
         DateTime.now().difference(lastGuidance!) > const Duration(seconds: 2))) {
      
      _isReady = false;
      lastMessage = message;
      lastGuidance = DateTime.now();
      
      await tts.speak(message);
      
      _cooldownTimer?.cancel();
      _cooldownTimer = Timer(const Duration(milliseconds: 500), () {
        _isReady = true;
      });
    }
  }

  String _determineGuidanceMessage(double deviation) {
    final absDeviation = deviation.abs();
    
    if (absDeviation < _CENTER_THRESHOLD) {
      return "Centered";
    } else if (absDeviation < _SLIGHT_THRESHOLD) {
      return deviation > 0 ? "Slight left" : "Slight right";
    } else if (absDeviation < _MODERATE_THRESHOLD) {
      return deviation > 0 ? "Move left" : "Move right";
    } else {
      // Add urgency for significant deviations
      return deviation > 0 ? "Far left!" : "Far right!";
    }
  }

  Future<void> stop() async {
    _cooldownTimer?.cancel();
    _isReady = true;
    lastMessage = null;
    lastGuidance = null;
    _lastDeviation = null;
    _consecutiveSimilarReadings = 0;
    await tts.stop();
  }

  void dispose() {
    _cooldownTimer?.cancel();
    tts.stop();
  }
} 