import 'package:flutter_tts/flutter_tts.dart';
import 'dart:async';

class GuidanceService {
  final FlutterTts tts = FlutterTts();
  DateTime? lastGuidance;
  String? lastMessage;
  Timer? _cooldownTimer;
  bool _isReady = true;

  GuidanceService() {
    _initTTS();
  }

  Future<void> _initTTS() async {
    await tts.setLanguage('en-US');
    await tts.setSpeechRate(0.9);  // Slower, clearer speech
    await tts.setVolume(1.0);
    await tts.setPitch(1.0);
  }

  Future<void> provideGuidance(double deviation) async {
    if (!_isReady) return;

    String message;
    if (deviation.abs() < 0.05) {
      message = "Centered";
    } else if (deviation.abs() < 0.2) {
      message = deviation > 0 ? "Slight left" : "Slight right";
    } else if (deviation.abs() < 0.4) {
      message = deviation > 0 ? "Move left" : "Move right";
    } else {
      message = deviation > 0 ? "Far left" : "Far right";
    }

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

  Future<void> stop() async {
    _cooldownTimer?.cancel();
    _isReady = true;
    lastMessage = null;
    lastGuidance = null;
    await tts.stop();
  }

  void dispose() {
    _cooldownTimer?.cancel();
    tts.stop();
  }
} 