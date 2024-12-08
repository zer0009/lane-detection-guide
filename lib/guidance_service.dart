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
    if (deviation.abs() < 0.1) {
      message = "Good, stay centered";
    } else if (deviation.abs() < 0.3) {
      message = deviation > 0 ? "Gently move left" : "Gently move right";
    } else if (deviation.abs() < 0.6) {
      message = deviation > 0 ? "Move more to the left" : "Move more to the right";
    } else {
      message = deviation > 0 ? "Warning, far left" : "Warning, far right";
    }

    // Only speak if message changed or enough time has passed
    if (message != lastMessage || 
        (lastGuidance != null && 
         DateTime.now().difference(lastGuidance!) > const Duration(seconds: 3))) {  // Increased delay
      
      _isReady = false;
      lastMessage = message;
      lastGuidance = DateTime.now();
      
      await tts.speak(message);
      
      // Increased cooldown for better spacing between messages
      _cooldownTimer?.cancel();
      _cooldownTimer = Timer(const Duration(milliseconds: 800), () {
        _isReady = true;
      });
    }
  }

  void dispose() {
    _cooldownTimer?.cancel();
    tts.stop();
  }
} 