import 'package:flutter/material.dart';
import 'dart:math' show pi;

class LaneGuidePainter extends CustomPainter {
  final double deviation;
  static const double CENTER_THRESHOLD = 0.05;
  static const double WARNING_THRESHOLD = 0.3;

  LaneGuidePainter(this.deviation);

  @override
  void paint(Canvas canvas, Size size) {
    final centerX = size.width / 2;
    final deviationX = centerX + (deviation * centerX);
    final isSignificantDeviation = deviation.abs() >= WARNING_THRESHOLD;
    
    // Draw guidance zone
    _drawGuidanceZone(canvas, size, centerX);
    
    // Draw center line
    _drawCenterLine(canvas, size, centerX);
    
    // Draw deviation line
    _drawDeviationLine(canvas, size, deviationX);
    
    // Draw guidance arrow if needed
    if (deviation.abs() >= CENTER_THRESHOLD) {
      _drawGuidanceArrow(canvas, size, centerX, isSignificantDeviation);
    }
    
    // Draw deviation indicator
    _drawDeviationIndicator(canvas, size, deviation);
  }

  void _drawGuidanceZone(Canvas canvas, Size size, double centerX) {
    final paint = Paint()
      ..color = Colors.green.withOpacity(0.1)
      ..style = PaintingStyle.fill;

    // Safe zone path
    final safeZonePath = Path()
      ..moveTo(centerX - (centerX * CENTER_THRESHOLD), 0)
      ..lineTo(centerX + (centerX * CENTER_THRESHOLD), 0)
      ..lineTo(centerX + (centerX * CENTER_THRESHOLD), size.height)
      ..lineTo(centerX - (centerX * CENTER_THRESHOLD), size.height)
      ..close();

    canvas.drawPath(safeZonePath, paint);
  }

  void _drawCenterLine(Canvas canvas, Size size, double centerX) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Dotted line effect
    final dashHeight = 10.0;
    final dashSpace = 5.0;
    var startY = 0.0;

    while (startY < size.height) {
      canvas.drawLine(
        Offset(centerX, startY),
        Offset(centerX, startY + dashHeight),
        paint,
      );
      startY += dashHeight + dashSpace;
    }
  }

  void _drawDeviationLine(Canvas canvas, Size size, double deviationX) {
    final paint = Paint()
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;

    // Color based on deviation severity
    if (deviation.abs() < CENTER_THRESHOLD) {
      paint.color = Colors.green.withOpacity(0.7);
    } else if (deviation.abs() < WARNING_THRESHOLD) {
      paint.color = Colors.yellow.withOpacity(0.7);
    } else {
      paint.color = Colors.red.withOpacity(0.7);
    }

    canvas.drawLine(
      Offset(deviationX, size.height),
      Offset(deviationX, size.height * 0.6),
      paint,
    );
  }

  void _drawGuidanceArrow(Canvas canvas, Size size, double centerX, bool isSignificant) {
    final arrowPaint = Paint()
      ..color = isSignificant ? Colors.red.withOpacity(0.8) : Colors.yellow.withOpacity(0.7)
      ..style = PaintingStyle.fill;

    final arrowPath = Path();
    final arrowY = size.height * 0.3;
    final arrowSize = isSignificant ? 40.0 : 30.0;
    final arrowThickness = isSignificant ? 25.0 : 20.0;

    if (deviation > 0) {
      // Arrow pointing left
      arrowPath
        ..moveTo(centerX - arrowSize, arrowY)
        ..lineTo(centerX + arrowThickness, arrowY - arrowThickness)
        ..lineTo(centerX + arrowThickness, arrowY + arrowThickness)
        ..close();
    } else {
      // Arrow pointing right
      arrowPath
        ..moveTo(centerX + arrowSize, arrowY)
        ..lineTo(centerX - arrowThickness, arrowY - arrowThickness)
        ..lineTo(centerX - arrowThickness, arrowY + arrowThickness)
        ..close();
    }

    // Add pulsing effect for significant deviations
    if (isSignificant) {
      final pulseValue = (DateTime.now().millisecondsSinceEpoch % 1000) / 1000;
      final opacity = 0.5 + (pulseValue * 0.5);
      arrowPaint.color = arrowPaint.color.withOpacity(opacity);
    }

    canvas.drawPath(arrowPath, arrowPaint);
  }

  void _drawDeviationIndicator(Canvas canvas, Size size, double deviation) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: '${(deviation * 100).toStringAsFixed(1)}%',
        style: TextStyle(
          color: deviation.abs() < CENTER_THRESHOLD 
              ? Colors.green 
              : deviation.abs() < WARNING_THRESHOLD
                  ? Colors.yellow
                  : Colors.red,
          fontSize: 24,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        (size.width - textPainter.width) / 2,
        size.height * 0.1,
      ),
    );
  }

  @override
  bool shouldRepaint(LaneGuidePainter oldDelegate) {
    return oldDelegate.deviation != deviation;
  }
} 